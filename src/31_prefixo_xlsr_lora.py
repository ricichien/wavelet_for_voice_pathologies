# -*- coding: utf-8 -*-
"""Prefixo XLS-R5 e LoRA nativa; importação/verificação sem Torch ou GPU.

O benchmark usa exclusivamente ruído e rótulos artificiais após a fila29–30.
Nenhum arquivo de modelo é modificado; estados treináveis são exportados à parte.
"""
from __future__ import annotations

import argparse
import copy
from functools import lru_cache
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SPEC = importlib.util.spec_from_file_location("exp20_for31", HERE / "20_extrair_wav2vec_avfad.py")
EXP20 = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(EXP20)
REVISION = "1a640f32ac3e39899438a2931f9924c02f080a54"
VERSION = "31.1"
PREFIX_BLOCKS, LORA_BLOCKS = 5, (3, 4)
RANK, ALPHA, LORA_DROPOUT = 4, 4., .05
OUT_REL = Path("results/experiments/resultados_31_benchmark_prefixo_lora")
EXPORTS = ("31_BENCHMARK_PREFIXO_LORA.json", "31_BENCHMARK_PREFIXO_LORA.txt", "assinatura_execucao.json", "PROTOCOLO.txt")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _torch():
    import torch
    return torch


@lru_cache(maxsize=1)
def _lora_type():
    torch = _torch(); nn = torch.nn
    class LoRALinear(nn.Module):
        def __init__(self, base, rank, alpha, dropout):
            super().__init__()
            require(isinstance(base, nn.Linear), "LoRA exige nn.Linear")
            require(1 <= rank <= min(base.in_features, base.out_features) and alpha > 0 and 0 <= dropout < 1,
                    "Configuração LoRA inválida")
            self.base = base.requires_grad_(False)
            self.rank, self.alpha, self.scaling = int(rank), float(alpha), float(alpha)/rank
            self.lora_A = nn.Linear(base.in_features, rank, bias=False, device=base.weight.device, dtype=base.weight.dtype)
            self.lora_B = nn.Linear(rank, base.out_features, bias=False, device=base.weight.device, dtype=base.weight.dtype)
            self.lora_dropout = nn.Dropout(dropout)
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)
            self.train(False)

        def forward(self, x):
            return self.base(x) + self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling

        def train(self, mode=True):
            self.training = bool(mode)
            self.base.eval()
            self.lora_A.train(mode); self.lora_B.train(mode); self.lora_dropout.train(mode)
            return self
    return LoRALinear


def lora_modules(prefix):
    return [(name, value) for name, value in prefix.named_modules() if isinstance(value, _lora_type())]


def adapter_names(prefix):
    return {f"{name}.{suffix}.weight" for name, _ in lora_modules(prefix) for suffix in ("lora_A", "lora_B")}


def install_lora(prefix, rank=RANK, alpha=ALPHA, dropout=LORA_DROPOUT, block_indices=LORA_BLOCKS):
    torch = _torch()
    require(len(prefix.encoder.layers) == PREFIX_BLOCKS and tuple(block_indices) == LORA_BLOCKS, "LoRA31 usa blocos3/4 de prefixo5")
    require(not lora_modules(prefix), "LoRA já instalada; não duplicar")
    before = frozen_weight_digest(prefix)
    prefix.requires_grad_(False)
    for index in block_indices:
        attention = prefix.encoder.layers[index].attention
        for projection in ("q_proj", "v_proj"):
            base = getattr(attention, projection)
            require(isinstance(base, torch.nn.Linear), "Projeção q/v incompatível")
            setattr(attention, projection, _lora_type()(base, rank, alpha, dropout))
    set_training(prefix, False)
    expected = adapter_names(prefix)
    require({name for name, value in prefix.named_parameters() if value.requires_grad} == expected, "Parâmetros treináveis fora da LoRA")
    require(frozen_weight_digest(prefix) == before, "Instalação LoRA alterou pesos base")
    return {"rank": rank, "alpha": float(alpha), "dropout": float(dropout), "block_indices": list(block_indices),
            "projections": ["q_proj", "v_proj"], "n_modules": len(lora_modules(prefix)),
            "n_trainable": sum(value.numel() for name, value in prefix.named_parameters() if name in expected)}


def set_training(prefix, training=True):
    """Desativa SpecAugment/LayerDrop/dropouts base; ativa só LoRA quando pedido."""
    prefix.eval()
    for _, value in lora_modules(prefix): value.train(training)
    return prefix


def assert_base_eval(prefix):
    require(not prefix.training, "Encoder deve ficar eval; usar set_training após wrapper.train")
    allowed = set()
    for name, value in lora_modules(prefix):
        allowed.update([name, name+".lora_A", name+".lora_B", name+".lora_dropout"])
    require(all(not value.training or name in allowed for name, value in prefix.named_modules()),
            "Módulo congelado em train; SpecAugment/LayerDrop não podem ser reativados")
    expected = adapter_names(prefix)
    require({name for name, value in prefix.named_parameters() if value.requires_grad} == expected,
            "Peso base treinável ou adaptador indevidamente congelado")


def prune_prefix(full_model, blocks=PREFIX_BLOCKS):
    """Muta somente este objeto em RAM; o caller pode passar deepcopy do full."""
    torch = _torch()
    require(blocks == PREFIX_BLOCKS and full_model.config.do_stable_layer_norm is True and
            len(full_model.encoder.layers) >= blocks, "Prefixo31 exige StableLayerNorm e cinco blocos")
    require(getattr(full_model, "adapter", None) is None, "Adapter original do checkpoint não é suportado neste prefixo")
    full_model.config = copy.deepcopy(full_model.config)
    full_model.encoder.layers = torch.nn.ModuleList(list(full_model.encoder.layers)[:blocks])
    full_model.encoder.layer_norm = torch.nn.Identity()
    full_model.config.num_hidden_layers = blocks
    full_model.requires_grad_(False); full_model.eval()
    return full_model


def copy_prefix(base, adapt=False):
    require(len(base.encoder.layers) == PREFIX_BLOCKS and not lora_modules(base), "Template deve ser prefixo5 sem adaptadores")
    result = copy.deepcopy(base).requires_grad_(False).eval()
    if adapt: install_lora(result)
    return result


def load_prefix(model_dir, device="cpu"):
    model_dir = Path(model_dir).resolve()
    files, use_safe = EXP20.model_files(model_dir, REVISION)
    torch, processor, model, runtime = EXP20.load_encoder(model_dir, use_safe, "cpu")
    model = prune_prefix(model)
    require(device in ("cpu", "cuda") and (device != "cuda" or torch.cuda.is_available()), "Dispositivo indisponível")
    model.to(device=device, dtype=torch.float32)
    runtime.update(device=device, gpu=torch.cuda.get_device_name(0) if device == "cuda" else None,
                   prefix_blocks=PREFIX_BLOCKS, final_layer_norm="Identity", model_revision=REVISION,
                   model_files={path.name: EXP20.sha(path) for path in files},
                   prefix_parameters=sum(value.numel() for value in model.parameters()))
    return processor, model, runtime


def pool_masked(prefix, hidden, attention_mask=None):
    """Média/DP populacional float32 de frames válidos, diferenciáveis e DP0 segura."""
    torch = _torch()
    require(hidden.ndim == 3 and hidden.shape[1] > 0 and torch.isfinite(hidden).all().item(), "Hidden states inválidos")
    if attention_mask is None:
        valid = torch.ones(hidden.shape[:2], device=hidden.device, dtype=torch.bool)
    else:
        require(attention_mask.ndim == 2 and attention_mask.shape[0] == hidden.shape[0], "Máscara de waveform incompatível")
        require(((attention_mask == 0) | (attention_mask == 1)).all().item() and
                (attention_mask[:, 1:] <= attention_mask[:, :-1]).all().item(), "Somente padding à direita binário é permitido")
        valid = prefix._get_feature_vector_attention_mask(hidden.shape[1], attention_mask).to(device=hidden.device, dtype=torch.bool)
    return pool_frames(hidden, valid)


def pool_frames(hidden, frame_mask=None):
    """Pooling quando a máscara já corresponde aos frames, sem downsample duplo."""
    torch = _torch()
    require(hidden.ndim == 3 and hidden.shape[1] > 0 and torch.isfinite(hidden).all().item(), "Hidden states inválidos")
    valid = torch.ones(hidden.shape[:2], device=hidden.device, dtype=torch.bool) if frame_mask is None else frame_mask.to(device=hidden.device, dtype=torch.bool)
    require(tuple(valid.shape) == tuple(hidden.shape[:2]) and valid.sum(dim=1).gt(0).all().item(), "Nenhum frame válido para pooling")
    mask = valid.unsqueeze(-1).to(hidden.dtype)
    count = mask.sum(dim=1)
    mean = (hidden*mask).sum(dim=1)/count
    variance = (((hidden-mean.unsqueeze(1))**2)*mask).sum(dim=1)/count
    positive = variance > 0
    safe = torch.sqrt(torch.where(positive, variance, torch.ones_like(variance)))
    std = torch.where(positive, safe, torch.zeros_like(safe))
    result = torch.cat([mean, std], dim=1)
    require(torch.isfinite(result).all().item(), "Pooling produziu valores não finitos")
    return result


def encode_pooled(prefix, input_values, attention_mask=None):
    # NÃO impor no_grad/inference_mode: os adaptadores precisam da retropropagação.
    assert_base_eval(prefix)
    output = prefix(input_values=input_values, attention_mask=attention_mask, output_hidden_states=False, return_dict=True)
    return pool_masked(prefix, output.last_hidden_state, attention_mask)


def encode_prefix3(prefix, input_values, attention_mask=None):
    """Frontend/posição UMA vez e blocos0..2 congelados; devolve H3 e máscara."""
    torch = _torch()
    assert_base_eval(prefix)
    require(len(prefix.encoder.layers) == 5 and isinstance(prefix.encoder.layer_norm, torch.nn.Identity), "Esperava prefixo5 sem LNfinal")
    extracted = prefix.feature_extractor(input_values).transpose(1, 2)
    frame_mask = None if attention_mask is None else prefix._get_feature_vector_attention_mask(extracted.shape[1], attention_mask, add_adapter=False)
    hidden, _ = prefix.feature_projection(extracted)
    hidden = prefix._mask_hidden_states(hidden, mask_time_indices=None, attention_mask=frame_mask)
    encoder = prefix.encoder
    if frame_mask is not None:
        hidden[~frame_mask.unsqueeze(-1).repeat(1, 1, hidden.shape[2])] = 0
    attention = encoder._update_full_mask(frame_mask, hidden)
    hidden = encoder.dropout(hidden + encoder.pos_conv_embed(hidden))
    for layer in encoder.layers[:3]:
        hidden = layer(hidden, attention_mask=attention, output_attentions=False)[0]
    return hidden, frame_mask


def encode_tail(prefix, h3, frame_mask=None):
    """Só blocos3/4; não repete posição, frontend, dropout ou LayerNormfinal."""
    torch = _torch()
    assert_base_eval(prefix)
    require(len(prefix.encoder.layers) == 5 and isinstance(prefix.encoder.layer_norm, torch.nn.Identity), "Esperava prefixo5 sem LNfinal")
    require(h3.ndim == 3 and torch.isfinite(h3).all().item(), "CacheH3 inválido")
    if torch.is_grad_enabled() and adapter_names(prefix):
        require(not torch.is_inference(h3), "CacheH3 para LoRA deve ser tensor regular, criado fora de inference_mode")
    if frame_mask is not None:
        require(tuple(frame_mask.shape) == tuple(h3.shape[:2]), "MáscaraH3 deve estar em frames")
    attention = prefix.encoder._update_full_mask(frame_mask, h3)
    hidden = h3
    for layer in prefix.encoder.layers[3:]:
        hidden = layer(hidden, attention_mask=attention, output_attentions=False)[0]
    return hidden


def encode_tail_pooled(prefix, h3, frame_mask=None):
    return pool_frames(encode_tail(prefix, h3, frame_mask), frame_mask)


def make_head(input_dim=2048, hidden=32, dropout=.25):
    nn = _torch().nn
    return nn.Sequential(nn.Linear(input_dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1))


def adapter_state(prefix):
    expected = adapter_names(prefix)
    return {name: value.detach().cpu().clone() for name, value in prefix.named_parameters() if name in expected}


def trainable_state(prefix):
    return adapter_state(prefix)


def load_adapter_state(prefix, state):
    torch = _torch(); target = dict(prefix.named_parameters()); expected = adapter_names(prefix)
    require(set(state) == expected, "Estado não contém exatamente os adaptadores esperados")
    before = frozen_weight_digest(prefix)
    with torch.no_grad():
        for name, value in state.items():
            require(tuple(value.shape) == tuple(target[name].shape) and torch.isfinite(value).all().item(), "Estado LoRA inválido")
            target[name].copy_(value.to(device=target[name].device, dtype=target[name].dtype))
    require(frozen_weight_digest(prefix) == before, "Restaurar adaptadores alterou a base")


def frozen_weight_digest(prefix):
    """Inclui toda base/buffers; remove sóA/B, canonicaliza projeção.base.*."""
    excluded = adapter_names(prefix)
    values = [(name, value) for name, value in prefix.named_parameters() if name not in excluded]
    values.extend(prefix.named_buffers())
    canonical = {}
    for name, value in values:
        key = name.replace(".base.", ".")
        require(key not in canonical, "Nome canônico duplicado no digest base")
        canonical[key] = value
    digest = hashlib.sha256()
    for name, value in sorted(canonical.items()):
        x = value.detach().cpu().contiguous()
        digest.update(name.encode("utf8")); digest.update(str(x.dtype).encode()); digest.update(str(tuple(x.shape)).encode())
        digest.update(x.numpy().tobytes())
    return digest.hexdigest()


def previous_complete(root):
    folder = root / "results/experiments/campanha_29_30_execucao"
    state = json.loads((folder / "STATUS.json").read_text(encoding="utf8"))
    require(state.get("phase") == "completo", "Benchmark31 aguarda término/auditoria29–30; nenhuma GPU iniciada")
    for number, name in ((29, "resultados_29_vogal_wavelet_ssl_extracao"), (30, "resultados_30_frase_vogal_wavelet")):
        marker = root / "results/experiments" / name / "ARTEFATOS_COMPLETOS.json"
        require(EXP20.sha(marker) == state.get(f"marker{number}_sha256"), "Marcador anterior diverge da fila")
        for key, digest in json.loads(marker.read_text(encoding="utf8")).items():
            require(Path(key).name == key and key not in (".", "..") and EXP20.sha(marker.parent/key) == digest, "Export anterior alterado")
    audit = root / "results/experiments/documentacao_wav2vec/30_VERIFICACAO_INDEPENDENTE.json"
    require(EXP20.sha(audit) == state.get("audit_sha256") and json.loads(audit.read_text(encoding="utf8")).get("status") == "aprovado",
            "Auditoria30 não aprovada/vinculada")


def benchmark_protocol():
    return """EXPERIMENTO31 — CUSTO SINTÉTICO DO PREFIXO XLS-R5 COM LoRA
Nenhum áudio clínico, rótulo real, classificador AVFAD/SVD ou acurácia medidos.
Modelo20 local/revisão fixa; full.hidden_states[5] deve igualar prefixo5 exatamente
antes de LoRA. Prefixo em RAM: cinco blocos e LayerNorm final trocada por Identity.
Base inteira congelada/eval: SpecAugment, LayerDrop e dropouts originais desativados.
LoRA nativa r4/alpha4/dropout0.05 em q_proj/v_proj dos blocos4/5, índices3/4.
A Kaiming e Bzero: identidade inicial; sóA/B e head são treináveis.
Pooling diferenciável média/DP populacional float32 de frames válidos; DP0 exata,
gradiente finito, sem epsilon adicionado ao valor. Head2048→32→1, dropout0.25.
Comparar frozenhead e LoRA+head em lotes1/2/4 pessoas×7gravações×5s16k.
Dados gaussianos artificiais, labels alternados inventados; média de7logits por
pessoa antes de BCE. Labels não são patologias; nenhuma métrica preditiva.
AdamW lr0.001/wd0.001;2passos aquecimento+10medidos; sincronização CUDA,
peak allocated/reserved VRAM, tempos por passo, grads adaptadores/head/base.
Frozenhead usa no_grad sóencoder, sem inference tensors; LoRA mantém autograd.
Float32, eager/TF32off/determinístico; preprocessing fora da medição de passos.
Digest completo da base antes/depois de12updates deve permanecer idêntico.
OOM por configuração é registrado, não causa alteração silenciosa de lote.
Escolha posterior de lote é operacional, antes de treino clínico e sem usar BA.
Só executar quando29–30 estiverem completos/auditados; não modificar suas fontes.
"""


def bench_case(base, processor, people, adapt, execution_mode="full_forward", seed=20261107):
    torch = _torch(); device = "cuda"
    torch.manual_seed(seed)
    model = copy_prefix(base, adapt=adapt).to(device)
    torch.manual_seed(seed)
    head = make_head(input_dim=2*base.config.hidden_size).to(device).train()
    set_training(model, adapt)
    before = frozen_weight_digest(model)
    rng = np.random.default_rng(seed)
    samples = [rng.normal(size=80000).astype(np.float32) for _ in range(people*7)]
    prepared = processor(samples, sampling_rate=16000, return_tensors="pt", return_attention_mask=True, padding=True)
    inputs = {key: value.to(device) for key, value in prepared.items()}
    require(execution_mode in ("full_forward", "tail_cached"), "Backend sintético desconhecido")
    global_cache, cached_frames = None, None
    if execution_mode == "tail_cached":
        with torch.no_grad(): h3, cached_frames = encode_prefix3(model, inputs["input_values"], inputs["attention_mask"])
        # Alocação normal, fora de inference_mode, representa2876gravações.
        global_cache = torch.empty((2876, h3.shape[1], h3.shape[2]), device=device, dtype=h3.dtype)
        global_cache[:len(h3)].copy_(h3)
        h3_input = global_cache[:len(h3)]
        require(not torch.is_inference(h3_input), "Cache sintético não pode ser inference tensor")
        del h3
    parameters = [value for value in model.parameters() if value.requires_grad] + list(head.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=.001, weight_decay=.001)
    labels = (torch.arange(people, device=device)%2).float()
    def step():
        optimizer.zero_grad(set_to_none=True)
        if adapt:
            pooled = encode_tail_pooled(model, h3_input, cached_frames) if execution_mode == "tail_cached" else encode_pooled(model, inputs["input_values"], inputs["attention_mask"])
        else:
            with torch.no_grad():
                pooled = encode_tail_pooled(model, h3_input, cached_frames) if execution_mode == "tail_cached" else encode_pooled(model, inputs["input_values"], inputs["attention_mask"])
        logits = head(pooled).reshape(people, 7).mean(dim=1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        require(torch.isfinite(loss).item(), "Loss sintética não finita")
        loss.backward(); optimizer.step()
    for _ in range(2): step()
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for _ in range(10): step()
    torch.cuda.synchronize(); elapsed = time.perf_counter()-started
    allocated, reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
    head_grad = sum(float(value.grad.abs().sum().detach().cpu()) for value in head.parameters() if value.grad is not None)
    adapter_grad = sum(float(value.grad.abs().sum().detach().cpu()) for name, value in model.named_parameters() if name in adapter_names(model) and value.grad is not None)
    require(head_grad > 0 and (not adapt or adapter_grad > 0), "Gradiente sintético ausente")
    require(all(value.grad is None for name, value in model.named_parameters() if name not in adapter_names(model)), "Base congelada recebeu gradiente")
    after = frozen_weight_digest(model); require(before == after, "Benchmark modificou pesos base")
    return {"people": people, "recordings": people*7, "mode": "lora_head" if adapt else "frozen_head", "execution_mode": execution_mode, "status": "ok",
            "seconds_10_steps": elapsed, "seconds_per_step": elapsed/10, "peak_allocated_bytes": allocated,
            "peak_reserved_bytes": reserved, "head_parameters": sum(p.numel() for p in head.parameters()),
            "adapter_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "head_gradient_abs_sum": head_grad, "adapter_gradient_abs_sum": adapter_grad,
            "synthetic_global_H3_records": 2876 if global_cache is not None else 0,
            "synthetic_global_H3_bytes": global_cache.numel()*global_cache.element_size() if global_cache is not None else 0,
            "frozen_weight_digest_before": before, "frozen_weight_digest_after": after}


def benchmark(root, out, model_dir):
    previous_complete(root)
    files, safe = EXP20.model_files(model_dir, REVISION)
    ident = {"version": VERSION, "model_id": EXP20.MODEL_ID, "revision": REVISION,
             "model_files": {p.name: EXP20.sha(p) for p in files}, "source_sha256": EXP20.sha(__file__),
             "source20_sha256": EXP20.sha(EXP20.__file__), "protocol_sha256": hashlib.sha256(benchmark_protocol().encode()).hexdigest(),
             "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
             "packages": {name: importlib.metadata.version(name) for name in ("numpy", "torch", "transformers", "safetensors")}}
    if (out / "ARTEFATOS_COMPLETOS.json").exists():
        require(json.loads((out / "assinatura_execucao.json").read_text(encoding="utf8")) == ident, "Assinatura31 mudou; preservar")
        marker = json.loads((out / "ARTEFATOS_COMPLETOS.json").read_text(encoding="utf8"))
        require(set(marker) == set(EXPORTS) and all(EXP20.sha(out/name) == digest for name, digest in marker.items()), "Exports31 alterados")
        print("Benchmark31 já completo; nenhuma GPU/medição repetida.", flush=True); return
    require(not out.exists() or not any(out.iterdir()), "Benchmark31 parcial preservado; não sobrescrever")
    out.mkdir(parents=True, exist_ok=True)
    EXP20.atomic_text(out / "assinatura_execucao.json", EXP20.json_text(ident))
    EXP20.atomic_text(out / "PROTOCOLO.txt", benchmark_protocol())
    torch, processor, full, runtime = EXP20.load_encoder(model_dir, safe, "cuda")
    rng = np.random.default_rng(31)
    data = processor(rng.normal(size=80000).astype(np.float32), sampling_rate=16000, return_tensors="pt", return_attention_mask=True, padding=False)
    data = {key: value.to("cuda") for key, value in data.items()}
    with torch.inference_mode():
        full_output = full(**data, output_hidden_states=True, return_dict=True)
        reference = full_output.hidden_states[5].cpu().clone()
        del full_output
    prefix = prune_prefix(full)
    with torch.inference_mode(): actual = prefix(**data, output_hidden_states=False, return_dict=True).last_hidden_state.cpu()
    error = float((actual-reference).abs().max()); require(torch.equal(actual, reference), "Prefixo não reproduz full.hidden_states[5] exatamente")
    prefix = prefix.cpu(); del data, reference, actual, full
    torch.cuda.empty_cache()
    results = []
    for execution_mode in ("full_forward", "tail_cached"):
        for people in (1, 2, 4):
            for adapt in (False, True):
                try:
                    result = bench_case(prefix, processor, people, adapt, execution_mode)
                except torch.cuda.OutOfMemoryError as exc:
                    result = {"people": people, "recordings": people*7, "mode": "lora_head" if adapt else "frozen_head", "execution_mode": execution_mode, "status": "out_of_memory", "error": str(exc)}
                results.append(result); torch.cuda.empty_cache()
                print(json.dumps(result, ensure_ascii=False), flush=True)
    require(any(row["status"] == "ok" and row["mode"] == "lora_head" for row in results), "Nenhum loteLoRA viável no benchmark")
    record = {"synthetic_only": True, "completed_at": EXP20.EXP18.timestamp(), "clinical_recordings": 0,
              "real_labels": 0, "runtime": runtime, "prefix_full_layer5_exact": True, "prefix_max_abs_error": error,
              "prefix_parameters": sum(value.numel() for value in prefix.parameters()), "cases": results,
              "source_sha256": EXP20.sha(__file__), "model_files": ident["model_files"]}
    EXP20.atomic_text(out / EXPORTS[0], EXP20.json_text(record))
    lines = ["BENCHMARK31 — PREFIXO5 + LoRA, SOMENTE DADOS ARTIFICIAIS", f"Concluído: {record['completed_at']}",
             f"Prefixo/fullL5: exato, erro={error}; parâmetros base={record['prefix_parameters']}",
             "12updates por caso,2aquecimento+10medidos. Nenhum áudio clínico, accuracy ou BA."]
    for row in results:
        lines.append(json.dumps(row, ensure_ascii=False, sort_keys=True))
    lines += ["", "Custos não incluem leitura/crop/WPT/processamento dos dados reais. Lote definido depois por custo, não por BA.",
              "Treino clínico/adaptação da AVFAD ainda não foram executados por este benchmark.", ""]
    EXP20.atomic_text(out / EXPORTS[1], "\n".join(lines))
    EXP20.atomic_text(out / "ARTEFATOS_COMPLETOS.json", EXP20.json_text({name: EXP20.sha(out/name) for name in EXPORTS}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--verificar", action="store_true"); actions.add_argument("--autoteste", action="store_true")
    actions.add_argument("--benchmark", action="store_true")
    parser.add_argument("--dados", type=Path, default=ROOT); parser.add_argument("--saida", type=Path); parser.add_argument("--modelo", type=Path)
    args = parser.parse_args()
    if args.autoteste:
        import unittest
        spec = importlib.util.spec_from_file_location("test_prefixo_lora31", HERE / "test_prefixo_lora31.py")
        tests = importlib.util.module_from_spec(spec); spec.loader.exec_module(tests)
        if not unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(tests)).wasSuccessful(): raise SystemExit(1)
        return
    root = args.dados.resolve(); model_dir = (args.modelo or root / "models/wav2vec2-xls-r-300m" / REVISION).resolve()
    EXP20.model_files(model_dir, REVISION)
    if args.verificar:
        print("ENTRADAS31 OK: modelo local/revisão verificados; nenhuma importaçãoTorch, GPU, WAV, escrita ou benchmark.", flush=True); return
    out = (args.saida or root / OUT_REL).resolve()
    require(out.is_relative_to(root / "results/experiments") and out != root / "results/experiments", "Saída31 deve ser subpasta de experiments")
    benchmark(root, out, model_dir)


if __name__ == "__main__": main()
