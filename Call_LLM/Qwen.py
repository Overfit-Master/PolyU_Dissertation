"""
此代码实现 Qwen 系列的api调用服务
"""


import os
from openai import OpenAI, AsyncOpenAI
from typing import List, Dict, Optional, Tuple, Union


class QwenClient:
    def __init__(self, key_name: str, conf_path: str = "api_key.conf"):
        """
        :param key_name: api_key.conf 中配置的标识名
        :param conf_path: 配置文件的本地路径，默认为当前目录的 api_key.conf
        """
        self.api_key = self._load_api_key(conf_path, key_name)
        self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

        # 初始化同步和异步客户端
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.async_client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    def _load_api_key(self, conf_path: str, key_name: str) -> str:
        """从本地文件读取并解析对应的 API Key"""
        if not os.path.exists(conf_path):
            raise FileNotFoundError(f"配置文件未找到: {conf_path}")

        with open(conf_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # 跳过空行和注释行
                if not line or line.startswith('#'):
                    continue

                # 按照第一个冒号进行分割
                if ':' in line:
                    k, v = line.split(':', 1)
                    if k.strip() == key_name:
                        return v.strip()

        raise ValueError(f"在 {conf_path} 中未找到名为 '{key_name}' 的 API Key 配置。")


    def _build_messages(self, prompt: str, system_prompt: Optional[str] = None) -> List[Dict[str, str]]:
        """构建基础的单轮对话 messages 结构"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages


    def generate(
            self,
            prompt: Union[str, List[Dict[str, str]]],
            system_prompt: Optional[str] = None,
            model: Optional[str] = None,
            return_usage: bool = False
    ) -> Union[str, Tuple[str, dict]]:
        """同步生成文本"""

        current_model = model
        messages = self._build_messages(prompt, system_prompt) if isinstance(prompt, str) else prompt

        response = self.client.chat.completions.create(
            model=current_model,
            messages=messages,
        )

        content = response.choices[0].message.content
        if return_usage:
            return content, (response.usage.model_dump() if response.usage else {})
        return content

    async def async_generate(
            self,
            prompt: Union[str, List[Dict[str, str]]],
            system_prompt: Optional[str] = None,
            model: Optional[str] = None,
            return_usage: bool = False
    ) -> Union[str, Tuple[str, dict]]:
        """异步生成文本"""

        current_model = model
        messages = self._build_messages(prompt, system_prompt) if isinstance(prompt, str) else prompt

        response = await self.async_client.chat.completions.create(
            model=current_model,
            messages=messages
        )

        content = response.choices[0].message.content
        if return_usage:
            return content, (response.usage.model_dump() if response.usage else {})
        return content