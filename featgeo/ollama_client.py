import requests
import time
from typing import List, Dict, Any, Optional


DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "qwen3:1.7b"


class OllamaClient:
    """Ollama client with an OpenAI-like interface."""
    
    def __init__(self, base_url: str = DEFAULT_OLLAMA_BASE_URL, model_name: str = DEFAULT_OLLAMA_MODEL):
        """Initialize the client."""
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name
        self.chat_url = f"{self.base_url}/api/chat"
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        max_tokens: int = 2048,
        n: int = 1,
        enable_thinking: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Return an OpenAI-compatible chat completion response."""
        responses = []
        total_attempts = 0
        max_total_attempts = n * 5
        
        while len(responses) < n and total_attempts < max_total_attempts:
            total_attempts += 1
            
            payload = {
                "model": self.model_name,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "num_predict": max_tokens,
                }
            }
            
            if enable_thinking:
                payload["enable_thinking"] = True
            
            if "min_p" in kwargs:
                payload["options"]["min_p"] = kwargs["min_p"]
            
            max_retries = 3
            success = False
            for attempt in range(max_retries):
                try:
                    resp = requests.post(self.chat_url, json=payload, timeout=120)
                    resp.raise_for_status()
                    result = resp.json()
                    
                    content = result.get("message", {}).get("content", "")
                    responses.append({
                        "message": {"content": content, "role": "assistant"},
                        "finish_reason": "stop"
                    })
                    success = True
                    break
                    
                except requests.exceptions.RequestException:
                    if attempt < max_retries - 1:
                        time.sleep(2)
            
            if not success:
                time.sleep(1)
        
        if len(responses) < n:
            raise Exception(
                f"[Ollama] Could not generate enough responses: requested {n}, "
                f"got {len(responses)} after {total_attempts} attempts."
            )
        
        return {
            "choices": responses,
            "model": self.model_name,
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }
    
    def test_connection(self) -> bool:
        """Test connectivity to the Ollama service."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            resp.raise_for_status()
            models = resp.json().get("models", [])
            model_names = [m.get("name") for m in models]
            
            if self.model_name not in model_names:
                print(f"[Ollama] Warning: model {self.model_name} was not found")
                print(f"[Ollama] Available models: {', '.join(model_names)}")
                return False
            
            print(f"[Ollama] Connection successful. Using model: {self.model_name}")
            return True
            
        except Exception as e:
            print(f"[Ollama] Connection failed: {e}")
            print(f"[Ollama] Please make sure the Ollama service is running: {self.base_url}")
            return False


_ollama_client: Optional[OllamaClient] = None


def get_ollama_client() -> OllamaClient:
    """Return the global Ollama client instance."""
    global _ollama_client
    
    if _ollama_client is None:
        try:
            from featgeo import config
            base_url = getattr(config, 'OLLAMA_BASE_URL', DEFAULT_OLLAMA_BASE_URL)
            model_name = getattr(config, 'OLLAMA_MODEL_NAME', DEFAULT_OLLAMA_MODEL)
        except ImportError:
            base_url = DEFAULT_OLLAMA_BASE_URL
            model_name = DEFAULT_OLLAMA_MODEL
        
        _ollama_client = OllamaClient(base_url=base_url, model_name=model_name)
    
    return _ollama_client

