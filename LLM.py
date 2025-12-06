import json
import requests
from pathlib import Path
import os

class LLMHyperParamAssistant:
    """
    一个简单的 LLM 超参数建议器：
    - memory = 当前目录下的关键 .py 文件 + 用户的文字描述
    - 调用任意 OpenAI 兼容的免费/低价 LLM 接口
    - 返回一个 JSON dict，用于指导 pred / cls / rl / lstm 的训练参数
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str = "gpt-4o-mini",
        max_context_chars: int = 6000,
        verbose: bool = True,      # 新增：是否打印调试信息
    ):
        self.api_key = api_key or os.getenv("LLM_API_KEY", "")
        self.base_url = base_url or os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
        self.model = model
        self.max_context_chars = max_context_chars
        self.verbose = verbose

    # ===== 内部工具方法 =====
    def _collect_code_memory(self, files: list[str] | None = None) -> str:
        root = Path(__file__).resolve().parent
        if files is None:
            files = [
                "Integrate_class.py",
                "LSTM_class.py",
                "CQL_class.py",
                "Classifier_class.py",
                "Self_training_class.py",
            ]
        parts: list[str] = []
        for fname in files:
            fpath = root / fname
            if not fpath.exists():
                continue
            try:
                text = fpath.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            snippet = text[: int(self.max_context_chars / max(len(files), 1))]
            parts.append(f"# File: {fname}\n{snippet}")
        joined = "\n\n".join(parts)
        return joined[: self.max_context_chars]

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """
        调用 OpenAI 兼容 chat completion 接口，并打印输入输出
        """
        if not self.api_key:
            raise RuntimeError("LLM_API_KEY is empty; set env var or pass api_key to LLMHyperParamAssistant.")

        url = f"{self.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "temperature": 0.1,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        if self.verbose:
            print("\n================ LLM REQUEST ================")
            print(f"[Model]      {self.model}")
            print(f"[Base URL]   {self.base_url}")
            print("---------- System prompt (trunc) ----------")
            print(system_prompt[:800])
            print("---------- User prompt (trunc) ------------")
            print(user_prompt[:800])
            print("===========================================\n")

        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        if self.verbose:
            print("============== LLM RAW RESPONSE ============")
            print(f"HTTP status: {resp.status_code}")
            # 打印前 1000 字符，避免太长
            print(resp.text[:1000])
            print("===========================================\n")

        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]

        if self.verbose:
            print("============== LLM MESSAGE CONTENT =========")
            print(content[:1000])
            print("===========================================\n")

        return content

    @staticmethod
    def _extract_json_blob(text: str) -> str:
        text = text.strip()
        if "```" in text:
            first = text.find("```")
            last = text.rfind("```")
            inner = text[first + 3 : last]
            inner = inner.lstrip("json").strip()
            text = inner

        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return text[start : end + 1]
        return text

    # ===== 对外主接口 =====
    def suggest_config(
                self,
                client_text: str,
                code_files: list[str] | None = None,
                data_summary: str | None = None,  # 可以传当前 returns/trends 摘要
        ) -> dict:
            code_context = self._collect_code_memory(code_files)
            system_prompt = (
                "You are an expert quantitative finance ML assistant. "
                "You will generate a COMPLETE hyperparameter JSON for all model trainers. "
                "Return ONLY a JSON object. No explanation. "
                "You must include all reasonable hyperparameters if applicable."
            )
            user_prompt = f"""
    Here is the current project code context (truncated):

    <<<CODE>>>
    {code_context}
    <<<END CODE>>>

    Here is a brief summary of the CURRENT DATASET used in this run:

    <<<DATA SUMMARY>>>
    {data_summary or "No dataset summary provided."}
    <<<END DATA SUMMARY>>>

    User request / high-level instructions:
    {client_text}

    You will output a JSON with all tunable hyperparameters for each trainer:

    1. LSTMRegressorTrainer ("lstm"):
       - enable          (bool, whether to train this block)
       - lr
       - max_epochs
       - train_ratio
       - hidden_dim1
       - hidden_dim2
       - batch_size

    2. TriRegressorTrainer ("pred"):
       - enable          (bool)
       - train_ratio
       - lr_params        (dict)
       - rf_params        (dict)
       - br_params        (dict)

    3. TriClassifierTrainer ("cls"):
       - enable          (bool)
       - train_ratio
       - xgb_params       (dict)
       - rf_params        (dict)

    4. CQLPolicyTrainer ("rl"):
       - enable          (bool)
       - train_ratio
       - n_steps
       - n_steps_per_epoch
       - batch_size
       - cql_config_kwargs (dict)

    Return JSON like:
    {{
      "lstm": {{
        "enable": true,
        "lr": 0.001, "max_epochs": 30, "train_ratio": 0.8,
        "hidden_dim1": 64, "hidden_dim2": 32,
        "batch_size": 64
      }},
      "pred": {{
        "enable": true,
        "train_ratio": 0.8,
        "lr_params": {{"fit_intercept": true}},
        "rf_params": {{"n_estimators": 500, "max_depth": 6}},
        "br_params": {{"alpha_1": 1e-6}}
      }},
      "cls": {{
        "enable": true,
        "train_ratio": 0.75,
        "xgb_params": {{"max_depth": 4, "n_estimators": 400}},
        "rf_params": {{"max_depth": 6}}
      }},
      "rl": {{
        "enable": true,
        "train_ratio": 0.8,
        "n_steps": 5000,
        "n_steps_per_epoch": 1000,
        "batch_size": 256,
        "cql_config_kwargs": {{"gamma": 0.99}}
      }}
    }}

    Return ONLY JSON.
    """

            try:
                raw = self._call_llm(system_prompt, user_prompt)
                blob = self._extract_json_blob(raw)
                cfg = json.loads(blob)
            except Exception as e:
                if self.verbose:
                    print("!!!! LLM suggest_config FAILED, fallback to empty cfg")
                    print("Error:", repr(e))
                cfg = {}

            def _num(d: dict, key: str, default):
                try:
                    v = d.get(key, default)
                    return float(v)
                except Exception:
                    return default

            def _bool(d: dict, key: str, default: bool = True) -> bool:
                if key not in d:
                    return default
                v = d[key]
                if isinstance(v, bool):
                    return v
                if isinstance(v, (int, float)):
                    return bool(v)
                if isinstance(v, str):
                    return v.strip().lower() in ["1", "true", "yes", "y"]
                return default

            lstm_cfg = cfg.get("lstm", {})
            pred_cfg = cfg.get("pred", {})
            cls_cfg = cfg.get("cls", {})
            rl_cfg = cfg.get("rl", {})

            out = {
                "lstm": {
                    "enable": _bool(lstm_cfg, "enable", True),
                    "lr": _num(lstm_cfg, "lr", 1e-3),
                    "max_epochs": int(_num(lstm_cfg, "max_epochs", 30)),
                    "train_ratio": _num(lstm_cfg, "train_ratio", 0.8),
                    "hidden_dim1": int(_num(lstm_cfg, "hidden_dim1", 64)),
                    "hidden_dim2": int(_num(lstm_cfg, "hidden_dim2", 32)),
                    "batch_size": int(_num(lstm_cfg, "batch_size", 64)),
                },
                "pred": {
                    "enable": _bool(pred_cfg, "enable", True),
                    "train_ratio": _num(pred_cfg, "train_ratio", 0.8),
                    "lr_params": pred_cfg.get("lr_params", {}),
                    "rf_params": pred_cfg.get("rf_params", {}),
                    "br_params": pred_cfg.get("br_params", {}),
                },
                "cls": {
                    "enable": _bool(cls_cfg, "enable", True),
                    "train_ratio": _num(cls_cfg, "train_ratio", 0.8),
                    "xgb_params": cls_cfg.get("xgb_params", {}),
                    "rf_params": cls_cfg.get("rf_params", {}),
                },
                "rl": {
                    "enable": _bool(rl_cfg, "enable", True),
                    "train_ratio": _num(rl_cfg, "train_ratio", 0.8),
                    "n_steps": int(_num(rl_cfg, "n_steps", 3000)),
                    "n_steps_per_epoch": int(_num(rl_cfg, "n_steps_per_epoch", 1000)),
                    "batch_size": int(_num(rl_cfg, "batch_size", 256)),
                    "cql_config_kwargs": rl_cfg.get("cql_config_kwargs", {}),
                },
            }

            if self.verbose:
                print("============== LLM FINAL CONFIG USED =======")
                import pprint
                pprint.pprint(out)
                print("===========================================\n")

            return out

