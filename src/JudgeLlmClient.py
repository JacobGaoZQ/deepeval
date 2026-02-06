from deepeval.models.base_model import DeepEvalBaseLLM
import httpx


class JudgeLlmClient(DeepEvalBaseLLM):
    def __init__(self, model_name, api_key, base_url, verify_ssl=False):
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.verify_ssl = verify_ssl

    def load_model(self):
        # 返回模型名称即可
        return self.model_name

    def generate(self, prompt: str) -> str:
        """同步生成方法，供 GEval 评估调用"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}]
        }

        # 使用同步 httpx 客户端，并设置 verify 选项
        with httpx.Client(verify=self.verify_ssl) as client:
            response = client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=120.0
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]

    async def a_generate(self, prompt: str) -> str:
        """异步生成方法（可选，部分异步 evaluate 流程会用到）"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}]
        }

        async with httpx.AsyncClient(verify=self.verify_ssl) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60.0
            )
            response.raise_for_status()
            res_json = response.json()
            return res_json["choices"][0]["message"]["content"]

    def get_model_name(self):
        return self.model_name
