from openai import OpenAI
import time
import json
import os
import re
import torch
from PIL import Image
import numpy as np
import base64
from .utils import generate_request_id, log_prepare, log_complete, log_error, ProgressBar

ENV_KEYS_API_KEY = ["COMFYUI_RN_API_KEY", "COMFLY_API_KEY", "RUNNODE_API_KEY", "RN_API_KEY", "LLM_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY"]
ENV_KEYS_BASE_URL = ["COMFYUI_RN_BASE_URL", "COMFLY_BASE_URL", "RUNNODE_BASE_URL", "RN_BASE_URL", "LLM_API_BASEURL", "OPENAI_BASE_URL", "OPENAI_API_BASE_URL", "DEEPSEEK_API_BASE_URL"]

MODEL_MAPPING = {
    "Kimi-K2.5": "moonshotai/Kimi-K2.5",
    "Kimi-K2-Thinking": "moonshotai/Kimi-K2-Thinking",
    "Qwen2.5-32B-Instruct": "Qwen/Qwen2.5-32B-Instruct",
    "Qwen2.5-72B-Instruct": "Qwen/Qwen2.5-72B-Instruct", 
    "Qwen2.5-VL-32B-Instruct": "Qwen/Qwen2.5-VL-32B-Instruct",
    "Qwen2.5-VL-72B-Instruct": "Qwen/Qwen2.5-VL-72B-Instruct",
    "Qwen3-32B": "Qwen/Qwen3-32B",
    "Qwen3-Coder-480B-A35B-Instruct": "Qwen/Qwen3-Coder-480B-A35B-Instruct",
    "Qwen3-VL-30B-A3B-Instruct": "Qwen/Qwen3-VL-30B-A3B-Instruct",
    "DeepSeek-R1": "deepseek-ai/DeepSeek-R1",
    "DeepSeek-R1-Distill-Qwen-32B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "DeepSeek-R1-0528": "deepseek-ai/DeepSeek-R1-0528",
    "DeepSeek-V3": "deepseek-ai/DeepSeek-V3",
    "DeepSeek-V3-0324": "deepseek-ai/DeepSeek-V3-0324",
    "DeepSeek-OCR": "deepseek-ai/DeepSeek-OCR",
    "GD-DeepSeek-R1": "GD/DeepSeek-R1",
    "glm-4-9b-chat": "glm-4-9b-chat",
    "GLM-4-32B-0414": "ZhipuAI/GLM-4-32B-0414",
    "GLM-4.1V-9B-Thinking": "ZhipuAI/GLM-4.1V-9B-Thinking",
    "PaddleOCR-VL-0.9B": "PaddleOCR-VL-0.9B",
}

def get_first_env(names):
    for n in names:
        v = os.environ.get(n)
        if v is not None and str(v).strip() != "":
            return v
    return None

def _normalize_baseurl(u):
    if not u:
        return u
    s = u.rstrip("/")
    if s.endswith("/v1") or "/v1/" in s:
        return s
    return s + "/v1"


def get_config():
    try:
        config_paths = [
            os.path.join(os.path.dirname(os.path.realpath(__file__)), "config", 'ComfyUI_rn_prompter-config.json'),
            os.path.join(os.path.dirname(os.path.realpath(__file__)), "config", 'comfyui_rn_prompter-config.json')
        ]

        config_path = None
        for path in config_paths:
            if os.path.exists(path):
                config_path = path
                break
        
        config = {}
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:  
                config = json.load(f)
        
        llm_config = config.get('llm', {})
        current_provider = llm_config.get('current_provider', 'quchi')
        providers = llm_config.get('providers', {})
        
        provider_config = providers.get(current_provider, {})

        def _env(names):
            for n in names:
                v = os.environ.get(n)
                if v is not None and str(v).strip() != "":
                    return v
            return None

        def _env_float(names, default):
            v = _env(names)
            if v is None:
                return default
            try:
                return float(v)
            except Exception:
                return default

        def _env_int(names, default):
            v = _env(names)
            if v is None:
                return default
            try:
                return int(v)
            except Exception:
                return default

        # 统一的 API Key 读取逻辑 (Env > Config)
        api_key = _env(["COMFYUI_RN_API_KEY", "COMFLY_API_KEY", "RUNNODE_API_KEY", "RN_API_KEY", "LLM_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY"]) or provider_config.get('api_key', '')
        base_url = _env(["COMFYUI_RN_BASE_URL", "COMFLY_BASE_URL", "RUNNODE_BASE_URL", "RN_BASE_URL", "LLM_API_BASEURL", "OPENAI_BASE_URL", "OPENAI_API_BASE_URL", "DEEPSEEK_API_BASE_URL"]) or provider_config.get('base_url', '')
        model = _env(["COMFYUI_RN_MODEL", "COMFLY_MODEL", "RUNNODE_MODEL", "RN_MODEL", "LLM_MODEL", "OPENAI_MODEL", "DEEPSEEK_MODEL"]) or provider_config.get('model', '')
        
        temperature = _env_float(["COMFYUI_RN_TEMPERATURE", "COMFLY_TEMPERATURE"], provider_config.get('temperature', 0.7))
        max_tokens = _env_int(["COMFYUI_RN_MAX_TOKENS", "COMFLY_MAX_TOKENS"], provider_config.get('max_tokens', 1000))
        top_p = _env_float(["COMFYUI_RN_TOP_P", "COMFLY_TOP_P"], provider_config.get('top_p', 0.9))

        return {
            'api_key': api_key,
            'model': model,
            'base_url': base_url,
            'temperature': temperature,
            'max_tokens': max_tokens,
            'top_p': top_p
        }
            
    except Exception as e:
        print(f"Error loading config: {e}")
        return {}


def save_config(config):
    # 禁用配置保存以防止密钥泄露
    # raise PermissionError("save_config is disabled")
    pass


def get_vllm_config():
    try:
        config_paths = [
            os.path.join(os.path.dirname(os.path.realpath(__file__)), "config", 'ComfyUI_rn_prompter-config.json'),
            os.path.join(os.path.dirname(os.path.realpath(__file__)), "config", 'comfyui_rn_prompter-config.json')
        ]

        config_path = None
        for path in config_paths:
            if os.path.exists(path):
                config_path = path
                break
        config = {}
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        vllm_config = config.get('vllm', {})
        current_provider = vllm_config.get('current_provider', 'quchi')
        providers = vllm_config.get('providers', {})
        provider_config = providers.get(current_provider, {})

        def _env(names):
            for n in names:
                v = os.environ.get(n)
                if v is not None and str(v).strip() != "":
                    return v
            return None

        def _env_float(names, default):
            v = _env(names)
            if v is None:
                return default
            try:
                return float(v)
            except Exception:
                return default

        def _env_int(names, default):
            v = _env(names)
            if v is None:
                return default
            try:
                return int(v)
            except Exception:
                return default

        api_key = _env(["COMFYUI_RN_API_KEY", "COMFLY_API_KEY", "RUNNODE_API_KEY", "RN_API_KEY", "LLM_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY"]) or provider_config.get('api_key', '')
        base_url = _env(["COMFYUI_RN_BASE_URL", "COMFLY_BASE_URL", "RUNNODE_BASE_URL", "RN_BASE_URL", "LLM_API_BASEURL", "OPENAI_BASE_URL", "OPENAI_API_BASE_URL", "DEEPSEEK_API_BASE_URL"]) or provider_config.get('base_url', '')
        model = _env(["COMFYUI_RN_MODEL", "COMFLY_MODEL", "RUNNODE_MODEL", "RN_MODEL", "LLM_MODEL", "OPENAI_MODEL", "DEEPSEEK_MODEL"]) or provider_config.get('model', '')
        temperature = _env_float(["COMFYUI_RN_TEMPERATURE", "COMFLY_TEMPERATURE"], provider_config.get('temperature', 0.7))
        max_tokens = _env_int(["COMFYUI_RN_MAX_TOKENS", "COMFLY_MAX_TOKENS"], provider_config.get('max_tokens', 1000))
        top_p = _env_float(["COMFYUI_RN_TOP_P", "COMFLY_TOP_P"], provider_config.get('top_p', 0.9))

        return {
            'api_key': api_key,
            'model': model,
            'base_url': base_url,
            'temperature': temperature,
            'max_tokens': max_tokens,
            'top_p': top_p
        }
    except Exception:
        return {}


def get_model_config(model_name):
    """
    根据模型显示名称查找配置文件中的配置
    
    Args:
        model_name: 用户在界面中选择的模型名称（如"Qwen2.5-VL-32B-Instruct"）
    
    Returns:
        包含该模型配置的字典，如果找不到则返回空字典
    """
    try:
        # 修改配置文件路径为新的配置文件
        config_paths = [
            os.path.join(os.path.dirname(os.path.realpath(__file__)), "config", 'ComfyUI_rn_prompter-config.json'),
            os.path.join(os.path.dirname(os.path.realpath(__file__)), "config", 'comfyui_rn_prompter-config.json')
        ]

        config_path = None
        for path in config_paths:
            if os.path.exists(path):
                config_path = path
                break
        
        config = {}
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:  
                config = json.load(f)
        
        # 模型名称映射字典（前端显示名称 -> 配置中的模型名称）
        mapping = MODEL_MAPPING
        
        # 获取映射后的模型名称
        mapped_name = mapping.get(model_name, model_name)
        
        # 直接在整个配置中查找匹配的model字段
        def find_matching_config(config_dict, target_model):
            """递归查找匹配的配置"""
            if isinstance(config_dict, dict):
                # 检查当前字典是否有model字段且匹配
                if 'model' in config_dict and config_dict['model'] == target_model:
                    # 找到了匹配的配置
                    return {
                        'api_key': config_dict.get('api_key', ''),
                        'model': config_dict.get('model', ''),
                        'base_url': config_dict.get('base_url', ''),
                        'temperature': config_dict.get('temperature', 0.7),
                        'max_tokens': config_dict.get('max_tokens', 1000),
                        'top_p': config_dict.get('top_p', 0.9)
                    }
                
                # 递归查找子字典
                for key, value in config_dict.items():
                    result = find_matching_config(value, target_model)
                    if result:
                        return result
            
            elif isinstance(config_dict, list):
                # 递归查找列表中的字典
                for item in config_dict:
                    result = find_matching_config(item, target_model)
                    if result:
                        return result
            
            return None
        
        # 在整个配置中查找
        result = find_matching_config(config, mapped_name)
        
        if result:
            print(f"Found config for {model_name} -> {mapped_name}")
            return result
        
        print(f"No matching config found for {model_name} -> {mapped_name}")
        return {}
            
    except Exception as e:
        print(f"Error loading model config: {e}")
        return {}


baseurl = get_config().get('base_url', '')


def encode_image_b64(ref_image):
    i = 255. * ref_image.cpu().numpy()[0]
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

    lsize = np.max(img.size)
    factor = 1
    while lsize / factor > 2048:
        factor *= 2
    img = img.resize((img.size[0] // factor, img.size[1] // factor))

    image_path = f'{time.time()}.webp'
    img.save(image_path, 'WEBP')

    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')

    os.remove(image_path)
    return base64_image


def _encode_frame_np_b64(frame_np, max_side=1024):
    if frame_np is None:
        return None
    if hasattr(frame_np, "dtype"):
        if frame_np.dtype != np.uint8:
            frame_np = frame_np.astype(np.float32)
            if frame_np.max() <= 1.0:
                frame_np = frame_np * 255.0
            frame_np = np.clip(frame_np, 0, 255).astype(np.uint8)
    if getattr(frame_np, "ndim", None) == 3:
        if frame_np.shape[0] in (1, 3, 4) and frame_np.shape[-1] not in (1, 3, 4):
            frame_np = np.transpose(frame_np, (1, 2, 0))
    frame_img = Image.fromarray(frame_np)
    lsize = np.max(frame_img.size)
    factor = 1
    while lsize / factor > max_side:
        factor *= 2
    if factor > 1:
        frame_img = frame_img.resize((frame_img.size[0] // factor, frame_img.size[1] // factor))
    from io import BytesIO
    buf = BytesIO()
    frame_img.save(buf, format="WEBP")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def extract_video_frames(video_tensor, max_frames=20):
    """
    从视频张量中提取关键帧
    
    Args:
        video_tensor: 视频张量，形状为 (frames, height, width, channels)
        max_frames: 最大提取帧数
    Returns:
        List[base64]: 编码后的图像列表
    """
    import torch
    
    frames = []
    total_frames = video_tensor.shape[0]
    
    # 计算采样间隔，均匀采样关键帧
    if total_frames <= max_frames:
        indices = range(total_frames)
    else:
        step = total_frames / max_frames
        indices = [int(i * step) for i in range(max_frames)]
    
    for idx in indices:
        # 获取单帧图像
        frame = video_tensor[idx]
        
        # 转换为合适的格式 (height, width, channels)
        if isinstance(frame, torch.Tensor):
            frame_np = frame.cpu().numpy()
        else:
            frame_np = frame
            
        frame_base64 = _encode_frame_np_b64(frame_np, max_side=1024)
        if frame_base64:
            frames.append(frame_base64)
    
    return frames


def extract_video_frames_from_source(video_source, max_frames=20):
    if video_source is None:
        return []
    if hasattr(video_source, "shape"):
        try:
            return extract_video_frames(video_source, max_frames=max_frames)
        except Exception:
            pass
    import io as _io
    if isinstance(video_source, _io.BytesIO):
        try:
            import av
            video_source.seek(0)
            with av.open(video_source, mode="r") as container:
                video_stream = next((s for s in container.streams if s.type == "video"), None)
                if video_stream is None:
                    return []
                total = int(getattr(video_stream, "frames", 0) or 0)
                indices = None
                if total > 0:
                    if total <= max_frames:
                        indices = set(range(total))
                    else:
                        step = total / max_frames
                        indices = set(int(i * step) for i in range(max_frames))
                frames = []
                for i, frame in enumerate(container.decode(video_stream)):
                    if indices is not None:
                        if i not in indices:
                            continue
                    else:
                        if max_frames and len(frames) >= max_frames:
                            break
                    arr = frame.to_ndarray(format="rgb24")
                    b64 = _encode_frame_np_b64(arr, max_side=1024)
                    if b64:
                        frames.append(b64)
                return frames
        except Exception:
            return []

    if not isinstance(video_source, str):
        raise TypeError(f"Unsupported video source type: {type(video_source)}")
    src = video_source.strip()
    if not src:
        return []

    import tempfile
    import urllib.parse
    import urllib.request

    if src.startswith("file://"):
        src = urllib.parse.unquote(src[len("file://"):])

    if len(src) >= 3 and src[1] == ":" and (src[2] == "\\" or src[2] == "/"):
        drive = src[0].lower()
        rest = src[2:].replace("\\", "/").lstrip("/")
        src = f"/mnt/{drive}/{rest}"

    local_path = src
    cleanup_path = None
    if src.startswith("http://") or src.startswith("https://"):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp.close()
        urllib.request.urlretrieve(src, tmp.name)
        local_path = tmp.name
        cleanup_path = tmp.name
    else:
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Video path not found: {local_path}")

    try:
        try:
            import cv2
            cap = cv2.VideoCapture(local_path)
            if not cap.isOpened():
                raise RuntimeError("cv2.VideoCapture failed")
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if total > 0:
                if total <= max_frames:
                    indices = list(range(total))
                else:
                    step = total / max_frames
                    indices = [int(i * step) for i in range(max_frames)]
                frames = []
                for idx in indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        continue
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    b64 = _encode_frame_np_b64(frame, max_side=1024)
                    if b64:
                        frames.append(b64)
                cap.release()
                return frames
            frames = []
            grabbed = 0
            want = max_frames
            stride = 1
            if want > 0:
                stride = max(1, int(30 / want))
            idx = 0
            while grabbed < want:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                if idx % stride == 0:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    b64 = _encode_frame_np_b64(frame, max_side=1024)
                    if b64:
                        frames.append(b64)
                        grabbed += 1
                idx += 1
            cap.release()
            return frames
        except Exception:
            import imageio.v3 as iio
            frames = []
            for i, frame in enumerate(iio.imiter(local_path)):
                if max_frames and len(frames) >= max_frames:
                    break
                b64 = _encode_frame_np_b64(frame, max_side=1024)
                if b64:
                    frames.append(b64)
            return frames
    finally:
        if cleanup_path:
            try:
                os.remove(cleanup_path)
            except Exception:
                pass


def _get_video_url(video):
    if video is None:
        return None
    stream_source = getattr(video, "get_stream_source", None)
    if callable(stream_source):
        try:
            v = stream_source()
            if isinstance(v, str) and v.strip():
                return v.strip()
            import io as _io
            if isinstance(v, _io.BytesIO):
                return v
        except Exception:
            pass
    if isinstance(video, dict):
        for k in ("video_url", "url", "uri", "path", "file_path", "filepath", "video_path", "filename"):
            v = video.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
    for attr in ("video_url", "url", "uri", "path", "file_path", "filepath", "video_path", "filename"):
        v = getattr(video, attr, None)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _get_video_tensor(video):
    if video is None:
        return None
    if isinstance(video, (list, tuple)) and len(video) > 0:
        video = video[0]
    if isinstance(video, dict):
        for k in ("frames", "tensor", "video", "data", "images", "image", "pixels", "frames_tensor", "frame"):
            v = video.get(k)
            if v is not None:
                return v
    for attr in ("frames", "tensor", "video", "data", "images", "image", "pixels", "frames_tensor", "frame"):
        v = getattr(video, attr, None)
        if v is not None:
            return v
    return video


class RN_Translator():
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "prompt"}),
                "advanced_options": (["auto", "force_english", "force_chinese"], {"default": "auto"}),
            },
            "optional": {
                "seed": ("INT", {"default": 100, "min": 0, "max": 0xffffffffffffffff}),
                "apiBaseUrl": ("STRING", {"default": ""}),
                "apiKey": ("STRING", {"default": ""}),
                "model": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("translation",)
    FUNCTION = "translate_text"
    CATEGORY = "RunNode/rn_prompter"
    TITLE = "RunNode Translator"

    def _split_text_into_chunks(self, text, max_chunk_size=400):
        if len(text) <= max_chunk_size:
            return [text]
        sentences = re.split(r'[.!?。！？]\s*', text)
        chunks = []
        current = ""
        for s in sentences:
            if len(current + s) <= max_chunk_size:
                current += (s + ". " if s else "")
            else:
                if current:
                    chunks.append(current.strip())
                current = (s + ". " if s else "")
        if current:
            chunks.append(current.strip())
        final = []
        for c in chunks:
            if len(c) <= max_chunk_size:
                final.append(c)
            else:
                for i in range(0, len(c), max_chunk_size):
                    final.append(c[i:i + max_chunk_size])
        return final

    def _detect_direction(self, text, advanced_options):
        if advanced_options == "force_English":
            return ("源语言", "English")
        if advanced_options == "force_Chinese":
            return ("source", "Chinese")
        if re.search(r"[\u4e00-\u9fff]", text):
            return ("中文", "English")
        return ("English", "Chinese")

    def _translate_chunk(self, chunk, src_label, dst_label, temperature=None, apiBaseUrl=None, apiKey=None, model=None):
        cfg = get_config()
        
        if apiKey and str(apiKey).strip() and str(apiKey) != "default":
            used_api_key = apiKey
        else:
            used_api_key = os.environ.get("COMFYUI_RN_API_KEY") or cfg.get("api_key", "")

        if apiBaseUrl and str(apiBaseUrl).strip() and str(apiBaseUrl) != "default":
            used_api_baseurl = _normalize_baseurl(apiBaseUrl)
        else:
            used_api_baseurl = _normalize_baseurl(os.environ.get("COMFYUI_RN_BASE_URL") or cfg.get("base_url") or "https://api.openai.com/v1")

        if model and str(model).strip() and str(model) != "default":
            used_model = model
        else:
            used_model = cfg.get("model") or "gpt-4o-mini"

        if not used_api_key:
            return "错误：请提供API密钥"

        try:
            client = OpenAI(api_key=used_api_key, base_url=used_api_baseurl)
            system_prompt = "你是一个专业的翻译助手，负责准确翻译文本内容。"
            user_prompt = f"""
                            请将以下{src_label}内容翻译成{dst_label}：
                            {chunk}
                            
                            要求：
                            - 保持原意不变
                            - 语言自然流畅
                            - 只返回翻译结果，不要添加任何解释或额外文本
                            """
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt.strip()},
            ]
            
            cfg_temperature = cfg.get("temperature")
            use_temperature = (
                temperature if temperature is not None else (
                    cfg_temperature if cfg_temperature is not None else 0.3
                )
            )
            
            cfg_max_tokens = cfg.get("max_tokens")
            use_max_tokens = cfg_max_tokens if cfg_max_tokens is not None else 512
            
            params = {
                "model": used_model,
                "messages": messages,
                "temperature": use_temperature,
                "max_tokens": use_max_tokens,
            }
            
            cfg_top_p = cfg.get("top_p")
            if cfg_top_p is not None:
                params["top_p"] = cfg_top_p
                
            completion = client.chat.completions.create(**params)
            if completion is not None and hasattr(completion, 'choices') and len(completion.choices) > 0:
                return completion.choices[0].message.content.strip()
            return "错误：API返回空结果"
        except Exception as e:
            return f"翻译错误：{str(e)}"

    def translate_text(self, prompt, advanced_options, seed=0, apiBaseUrl="default", apiKey="default", model="default"):
        request_id = generate_request_id("translate", "rn_prompter")
        log_prepare("文本翻译", request_id, "RunNode/Prompter-", "Prompter")
        rn_pbar = ProgressBar(request_id, "Prompter", streaming=True, task_type="文本翻译", source="RunNode/Prompter-")
        cleaned = re.sub(r'[\x00\x01-\x08\x0b\x0c\x0e-\x1f\x7f]', '', prompt or "")
        if not cleaned.strip():
            rn_pbar.error("错误：请输入要翻译的文本")
            return ("错误：请输入要翻译的文本",)
        src_label, dst_label = self._detect_direction(cleaned, advanced_options)
        chunks = self._split_text_into_chunks(cleaned, max_chunk_size=400)
        if len(chunks) == 1:
            res = self._translate_chunk(chunks[0], src_label, dst_label, temperature=None, apiBaseUrl=apiBaseUrl, apiKey=apiKey, model=model)
            if isinstance(res, str) and (res.startswith("错误：") or res.startswith("翻译错误")):
                rn_pbar.error(res)
            else:
                rn_pbar.done(char_count=len(res or ""))
            return (res,)
        translated = []
        for c in chunks:
            translated.append(self._translate_chunk(c, src_label, dst_label, temperature=None, apiBaseUrl=apiBaseUrl, apiKey=apiKey, model=model))
        joined = ' '.join(translated)
        if joined.startswith("错误：") or joined.startswith("翻译错误"):
            rn_pbar.error(joined)
        else:
            rn_pbar.done(char_count=len(joined))
        return (joined,)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float(time.time())


class RN_Prompt_Translator():
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "advanced_options": (["auto", "force_English", "force_Chinese"], {"default": "auto"}),
            },
            "optional": {
                "seed": ("INT", {"default": 100, "min": 0, "max": 0xffffffffffffffff}),
                "apiBaseUrl": ("STRING", {"default": ""}),
                "apiKey": ("STRING", {"default": ""}),
                "model": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("translation",)
    FUNCTION = "translate_text"
    CATEGORY = "RunNode/rn_prompter"
    TITLE = "RunNode Prompt Translator"

    def _split_text_into_chunks(self, text, max_chunk_size=400):
        if len(text) <= max_chunk_size:
            return [text]
        sentences = re.split(r'[.!?。！？]\s*', text)
        chunks = []
        current = ""
        for s in sentences:
            if len(current + s) <= max_chunk_size:
                current += (s + ". " if s else "")
            else:
                if current:
                    chunks.append(current.strip())
                current = (s + ". " if s else "")
        if current:
            chunks.append(current.strip())
        final = []
        for c in chunks:
            if len(c) <= max_chunk_size:
                final.append(c)
            else:
                for i in range(0, len(c), max_chunk_size):
                    final.append(c[i:i + max_chunk_size])
        return final

    def _detect_direction(self, text, advanced_options):
        if advanced_options == "force_English":
            return ("源语言", "English")
        if advanced_options == "force_Chinese":
            return ("source", "Chinese")
        if re.search(r"[\u4e00-\u9fff]", text):
            return ("中文", "English")
        return ("English", "Chinese")

    def _translate_chunk(self, chunk, src_label, dst_label, temperature=None, apiBaseUrl=None, apiKey=None, model=None):
        cfg = get_config()
        
        if apiKey and str(apiKey).strip() and str(apiKey) != "default":
            used_api_key = apiKey
        else:
            used_api_key = os.environ.get("COMFYUI_RN_API_KEY") or cfg.get("api_key", "")

        if apiBaseUrl and str(apiBaseUrl).strip() and str(apiBaseUrl) != "default":
            used_api_baseurl = _normalize_baseurl(apiBaseUrl)
        else:
            used_api_baseurl = _normalize_baseurl(os.environ.get("COMFYUI_RN_BASE_URL") or cfg.get("base_url") or "https://api.openai.com/v1")

        if model and str(model).strip() and str(model) != "default":
            used_model = model
        else:
            used_model = cfg.get("model") or "gpt-4o-mini"

        if not used_api_key:
            return "错误：请提供API密钥"

        try:
            client = OpenAI(api_key=used_api_key, base_url=used_api_baseurl)
            system_prompt = "你是资深提示词工程师，负责将输入内容重写为用于生成式模型的标准化提示词，保持简洁、具象、可执行。"
            user_prompt = f"""
                            Rewrite the following content as a standard {dst_label} prompt for generative models.
                            Requirements:
                            - concise, vivid, comma-separated phrases
                            - include subject, attributes, composition, style, lighting
                            - no explanations, headers or extra text
                            - return only the final prompt in {dst_label}
                            Content:
                            {chunk}
                            """
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt.strip()},
            ]
            
            cfg_temperature = cfg.get("temperature")
            use_temperature = (
                temperature if temperature is not None else (
                    cfg_temperature if cfg_temperature is not None else 0.3
                )
            )
            
            cfg_max_tokens = cfg.get("max_tokens")
            use_max_tokens = cfg_max_tokens if cfg_max_tokens is not None else 512
            
            params = {
                "model": used_model,
                "messages": messages,
                "temperature": use_temperature,
                "max_tokens": use_max_tokens,
            }
            
            cfg_top_p = cfg.get("top_p")
            if cfg_top_p is not None:
                params["top_p"] = cfg_top_p
                
            completion = client.chat.completions.create(**params)
            if completion is not None and hasattr(completion, 'choices') and len(completion.choices) > 0:
                return completion.choices[0].message.content.strip()
            return "错误：API返回空结果"
        except Exception as e:
            return f"翻译错误：{str(e)}"

    def translate_text(self, prompt, advanced_options, seed=0, apiBaseUrl="default", apiKey="default", model="default"):
        request_id = generate_request_id("prompt_translate", "rn_prompter")
        log_prepare("提示词翻译", request_id, "RunNode/Prompter-", "Prompter")
        rn_pbar = ProgressBar(request_id, "Prompter", streaming=True, task_type="提示词翻译", source="RunNode/Prompter-")
        cleaned = re.sub(r'[\x00\x01-\x08\x0b\x0c\x0e-\x1f\x7f]', '', prompt or "")
        if not cleaned.strip():
            rn_pbar.error("错误：请输入要翻译的文本")
            return ("错误：请输入要翻译的文本",)
        src_label, dst_label = self._detect_direction(cleaned, advanced_options)
        chunks = self._split_text_into_chunks(cleaned, max_chunk_size=400)
        if len(chunks) == 1:
            res = self._translate_chunk(chunks[0], src_label, dst_label, temperature=None, apiBaseUrl=apiBaseUrl, apiKey=apiKey, model=model)
            if isinstance(res, str) and (res.startswith("错误：") or res.startswith("翻译错误")):
                rn_pbar.error(res)
            else:
                rn_pbar.done(char_count=len(res or ""))
            return (res,)
        translated = []
        for c in chunks:
            translated.append(self._translate_chunk(c, src_label, dst_label, temperature=None, apiBaseUrl=apiBaseUrl, apiKey=apiKey, model=model))
        joined = ' '.join(translated)
        if joined.startswith("错误：") or joined.startswith("翻译错误"):
            rn_pbar.error(joined)
        else:
            rn_pbar.done(char_count=len(joined))
        return (joined,)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float(time.time())


class RN_Midjourney_Prompter():
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "seed": ("INT", {"default": 100, "min": 0, "max": 0xffffffffffffffff}),
                "temperature": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 2.0, "step": 0.1}),
                "apiBaseUrl": ("STRING", {"default": "default"}),
                "apiKey": ("STRING", {"default": "default"}),
                "model": ("STRING", {"default": "default"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt opt",)
    FUNCTION = "generate_mj_prompt"
    CATEGORY = "RunNode/rn_prompter"
    TITLE = "Midjourney Style Prompter"

    def _split_text_into_chunks(self, text, max_chunk_size=400):
        if len(text) <= max_chunk_size:
            return [text]
        sentences = re.split(r'[.!?。！？]\s*', text)
        chunks = []
        current = ""
        for s in sentences:
            if len(current + s) <= max_chunk_size:
                current += (s + ". " if s else "")
            else:
                if current:
                    chunks.append(current.strip())
                current = (s + ". " if s else "")
        if current:
            chunks.append(current.strip())
        final = []
        for c in chunks:
            if len(c) <= max_chunk_size:
                final.append(c)
            else:
                for i in range(0, len(c), max_chunk_size):
                    final.append(c[i:i + max_chunk_size])
        return final

    def _generate_chunk_mj(self, chunk, temperature=None, apiBaseUrl=None, apiKey=None, model=None):
        cfg = get_config()
        
        # 处理输入回退逻辑: Input -> Env -> Config
        if apiKey and str(apiKey).strip() and str(apiKey) != "default":
            used_api_key = apiKey
        else:
            used_api_key = cfg.get("api_key", "")

        if apiBaseUrl and str(apiBaseUrl).strip() and str(apiBaseUrl) != "default":
            used_api_baseurl = apiBaseUrl
        else:
            used_api_baseurl = cfg.get("base_url") or "https://ai.t8star.cn//v1"

        if model and str(model).strip() and str(model) != "default":
            used_model = model
        else:
            used_model = cfg.get("model") or "gpt-4o-mini"
            
        if not used_api_key:
            return "错误：请提供API密钥"

        try:
            client = OpenAI(api_key=used_api_key, base_url=used_api_baseurl)
            system_prompt = "You are a professional Midjourney prompt engineer."
            user_prompt = f"""
                            Rewrite the following description into a high-quality Midjourney prompt.
                            - Format: [Subject], [Action/Context], [Art Style/Medium], [Lighting], [Colors], [Composition], --ar 16:9 --v 6.0
                            - Keep it descriptive but concise.
                            - Use English only.
                            Content:
                            {chunk}
                            """
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt.strip()},
            ]
            
            cfg_temperature = cfg.get("temperature")
            use_temperature = (
                temperature if temperature is not None else (
                    cfg_temperature if cfg_temperature is not None else 0.3
                )
            )
            
            cfg_max_tokens = cfg.get("max_tokens")
            use_max_tokens = cfg_max_tokens if cfg_max_tokens is not None else 512
            
            params = {
                "model": used_model,
                "messages": messages,
                "temperature": use_temperature,
                "max_tokens": use_max_tokens,
            }
            
            cfg_top_p = cfg.get("top_p")
            if cfg_top_p is not None:
                params["top_p"] = cfg_top_p
                
            completion = client.chat.completions.create(**params)
            if completion is not None and hasattr(completion, 'choices') and len(completion.choices) > 0:
                return completion.choices[0].message.content.strip()
            return "Error: Empty response"
        except Exception as e:
            return f"Error: {str(e)}"

    def generate_mj_prompt(self, prompt, seed=0, temperature=0.3, apiBaseUrl="default", apiKey="default", model="default"):
        request_id = generate_request_id("mj_prompt", "rn_prompter")
        log_prepare("MJ 提示词", request_id, "RunNode/Prompter-", "Prompter")
        rn_pbar = ProgressBar(request_id, "Prompter", streaming=True, task_type="MJ提示词", source="RunNode/Prompter-")
        cleaned = re.sub(r'[\x00\x01-\x08\x0b\x0c\x0e-\x1f\x7f]', '', prompt or "")
        if not cleaned.strip():
            rn_pbar.error("错误：请输入要生成的文本")
            return ("错误：请输入要生成的文本",)
        chunks = self._split_text_into_chunks(cleaned, max_chunk_size=400)
        if len(chunks) == 1:
            res = self._generate_chunk_mj(chunks[0], temperature=temperature, apiBaseUrl=apiBaseUrl, apiKey=apiKey, model=model)
            if isinstance(res, str) and (res.startswith("错误：") or res.startswith("Error")):
                rn_pbar.error(res)
            else:
                rn_pbar.done(char_count=len(res or ""))
            return (res,)
        generated = []
        for c in chunks:
            generated.append(self._generate_chunk_mj(c, temperature=temperature, apiBaseUrl=apiBaseUrl, apiKey=apiKey, model=model))
        joined = ' '.join(generated)
        if joined.startswith("错误：") or joined.startswith("Error"):
            rn_pbar.error(joined)
        else:
            rn_pbar.done(char_count=len(joined))
        return (joined,)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float(time.time())


class RN_LLMAPI_Node():
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_baseurl": ("STRING", {"multiline": True}),
                "api_key": ("STRING", {"default": ""}),
                "model": ("STRING", {"default": ""}),
                "role": ("STRING", {"multiline": True, "default": "You are a helpful assistant"}),
                "prompt": ("STRING", {"multiline": True, "default": "Hello"}),
                "temperature": ("FLOAT", {"default": 0.6}),
                "seed": ("INT", {"default": 100}),
                # "max_video_frames": ("INT", {"default": 5, "min": 1, "max": 20, "step": 1}),
            },
            "optional": {
                "ref_image": ("IMAGE",),
                "video": ("VIDEO",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("describe",)
    FUNCTION = "rn_run_llmapi"
    CATEGORY = "RunNode/rn_prompter"

    def rn_run_llmapi(self, api_baseurl, api_key, model, role, prompt, temperature, seed, 
                      max_video_frames=5, ref_image=None, video=None):
        request_id = generate_request_id("llmapi", "rn_prompter")
        log_prepare("通用LLM调用", request_id, "RunNode/Prompter-", "Prompter")
        rn_pbar = ProgressBar(request_id, "Prompter", streaming=True, task_type="LLM调用", source="RunNode/Prompter-")
        cfg = get_config()
        vllm_cfg = get_vllm_config()
        has_visual = (ref_image is not None) or (video is not None)
        selected_cfg = vllm_cfg if has_visual else cfg
        
        # 处理输入回退逻辑：只有当输入不为空（且不全是空格）时才使用输入值
        used_api_baseurl = api_baseurl if api_baseurl and api_baseurl.strip() else None
        used_api_key = api_key if api_key and api_key.strip() else None
        
        used_api_baseurl = _normalize_baseurl(used_api_baseurl or os.environ.get("COMFYUI_RN_BASE_URL") or selected_cfg.get("base_url"))
        used_api_key = (used_api_key or os.environ.get("COMFYUI_RN_API_KEY") or selected_cfg.get("api_key") or "")
        used_model = (model or selected_cfg.get("model") or ("qwen25-vl-32b-instruct" if has_visual else "gpt-4o-mini"))
        
        client = OpenAI(api_key=used_api_key, base_url=used_api_baseurl)
        
        messages = [{'role': 'system', 'content': f'{role}'}]
        user_content = [{"type": "text", "text": f"{prompt}"}]
        
        # 处理图片输入
        if ref_image is not None:
            base64_image = encode_image_b64(ref_image)
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/webp;base64,{base64_image}",
                    "detail": "high"
                }
            })
        
        # 处理视频输入
        if video is not None:
            try:
                video_frames = []
                get_components = getattr(video, "get_components", None)
                if callable(get_components):
                    try:
                        components = get_components()
                        images = getattr(components, "images", None)
                        if images is not None:
                            video_frames = extract_video_frames(images, max_frames=max_video_frames)
                    except Exception:
                        pass
                if not video_frames:
                    video_tensor = _get_video_tensor(video)
                    if hasattr(video_tensor, "shape"):
                        video_frames = extract_video_frames(video_tensor, max_frames=max_video_frames)
                if not video_frames:
                    video_src = _get_video_url(video)
                    if video_src:
                        video_frames = extract_video_frames_from_source(video_src, max_frames=max_video_frames)
                for frame_base64 in video_frames:
                    user_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/webp;base64,{frame_base64}",
                            "detail": "auto"
                        }
                    })
                if len(video_frames) > 0:
                    prompt_addition = f"\nNote: I've provided {len(video_frames)} keyframes from the video. "
                    user_content[0]["text"] = prompt + prompt_addition
                else:
                    video_src = _get_video_url(video)
                    video_tensor = _get_video_tensor(video)
                    tensor_shape = getattr(video_tensor, "shape", None)
                    keys = list(video.keys())[:30] if isinstance(video, dict) else None
                    rn_pbar.error(f"视频解析失败：未能从 VIDEO 中获得可用帧（type={type(video)} src={video_src} tensor_type={type(video_tensor)} tensor_shape={tensor_shape} keys={keys}）")
            except Exception as e:
                rn_pbar.error(f"视频处理出错: {str(e)}")
        
        # 构建消息
        messages.append({
            'role': 'user',
            'content': user_content
        })
        
        try:
            completion = client.chat.completions.create(
                model=used_model, 
                messages=messages, 
                temperature=temperature,
                seed=seed
            )
            if completion is not None and hasattr(completion, 'choices'):
                response = completion.choices[0].message.content
            else:
                response = 'Error: No response from API'
        except Exception as e:
            response = f"API调用出错: {str(e)}"
        if isinstance(response, str) and (response.startswith("Error") or response.startswith("API调用出错")):
            rn_pbar.error(response)
        else:
            rn_pbar.done(char_count=len(response or ""))
        return (response,)


class RN_LLMAPI_Pro_Node():
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (list(MODEL_MAPPING.keys()), {"default": "Kimi-K2.5"}),
                "role": ("STRING", {"multiline": True, "default": "You are a helpful assistant"}),
                "prompt": ("STRING", {"multiline": True, "default": "Hello"}),
                "temperature": ("FLOAT", {"default": 0.6}),
                "seed": ("INT", {"default": -1}),
            },
            "optional": {
                "ref_image1": ("IMAGE",),
                "ref_image2": ("IMAGE",),
                "ref_image3": ("IMAGE",),
                "ref_image4": ("IMAGE",),
                "ref_image5": ("IMAGE",),
                "ref_image6": ("IMAGE",),
                "ref_image7": ("IMAGE",),
                "ref_image8": ("IMAGE",),
                "ref_image9": ("IMAGE",),
                "video": ("VIDEO",),
                "max_video_frames": ("INT", {"default": 5, "min": 1, "max": 20, "step": 1}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "rn_run_llmapi_pro"
    CATEGORY = "RunNode/rn_prompter"

    def rn_run_llmapi_pro(
        self,
        model,
        role,
        prompt,
        temperature,
        seed,
        api_baseurl='',
        api_key='',
        ref_image1=None,
        ref_image2=None,
        ref_image3=None,
        ref_image4=None,
        ref_image5=None,
        ref_image6=None,
        ref_image7=None,
        ref_image8=None,
        ref_image9=None,
        video=None,
        max_video_frames=5,
    ):
        request_id = generate_request_id("llmapi_pro", "rn_prompter")
        log_prepare("专业LLM调用", request_id, "RunNode/Prompter-", "Prompter")
        rn_pbar = ProgressBar(request_id, "Prompter", streaming=True, task_type="LLM调用", source="RunNode/Prompter-")
        
        # 获取环境变量
        env_base_url = get_first_env(ENV_KEYS_BASE_URL)
        env_api_key = get_first_env(ENV_KEYS_API_KEY)

        # 获取模型对应的配置
        model_config = get_model_config(model)
        
        # 处理输入回退逻辑：只有当输入不为空（且不全是空格）时才使用输入值
        used_api_baseurl = api_baseurl if api_baseurl and api_baseurl.strip() else None
        used_api_key = api_key if api_key and api_key.strip() else None
        
        # 优先级: 输入参数 > 环境变量 > 模型特定配置 > 全局配置
        used_api_baseurl = used_api_baseurl or env_base_url or model_config.get('base_url', '') or get_config().get('base_url', '')
        used_api_key = used_api_key or env_api_key or model_config.get('api_key', '') or get_config().get('api_key', '')
        used_model = model_config.get('model', '')  # 使用配置中的模型名称
        
        # 如果配置中没有模型名称，使用映射后的名称
        if not used_model:
            # 模型名称映射字典
            mapping = MODEL_MAPPING
            used_model = mapping.get(model, model)
        
        # 最终的检查，确保有base_url
        if not used_api_baseurl:
            return ("ERROR: No API base URL configured. Please check your configuration.",)
        
        used_api_baseurl = _normalize_baseurl(used_api_baseurl)
        print(f"Using base_url: {used_api_baseurl}")
        print(f"Using model: {used_model}")
        
        client = OpenAI(api_key=used_api_key, base_url=used_api_baseurl)
        
        images_in = [ref_image1, ref_image2, ref_image3, ref_image4, ref_image5, ref_image6, ref_image7, ref_image8, ref_image9]
        image_batches = [img for img in images_in if img is not None]
        has_visual = bool(image_batches) or (video is not None)

        if not has_visual:
            messages = [
                {'role': 'system', 'content': f'{role}'},
                {'role': 'user', 'content': f'{prompt}'},
            ]
        else:
            user_content = [{"type": "text", "text": f"{prompt}"}]

            max_images = 9
            added_images = 0
            for img in image_batches:
                if added_images >= max_images:
                    break
                try:
                    batch_size = int(getattr(img, "shape", [0])[0]) if getattr(img, "shape", None) is not None else 0
                except Exception:
                    batch_size = 0
                if batch_size <= 1:
                    base64_image = encode_image_b64(img)
                    user_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/webp;base64,{base64_image}",
                            "detail": "high"
                        }
                    })
                    added_images += 1
                else:
                    for i in range(batch_size):
                        if added_images >= max_images:
                            break
                        base64_image = encode_image_b64(img[i:i+1])
                        user_content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/webp;base64,{base64_image}",
                                "detail": "high"
                            }
                        })
                        added_images += 1

            if video is not None:
                try:
                    video_frames = []
                    get_components = getattr(video, "get_components", None)
                    if callable(get_components):
                        try:
                            components = get_components()
                            images = getattr(components, "images", None)
                            if images is not None:
                                video_frames = extract_video_frames(images, max_frames=max_video_frames)
                        except Exception:
                            pass
                    if not video_frames:
                        video_tensor = _get_video_tensor(video)
                        if hasattr(video_tensor, "shape"):
                            video_frames = extract_video_frames(video_tensor, max_frames=max_video_frames)
                    if not video_frames:
                        video_src = _get_video_url(video)
                        if video_src:
                            video_frames = extract_video_frames_from_source(video_src, max_frames=max_video_frames)
                    for frame_base64 in video_frames:
                        user_content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/webp;base64,{frame_base64}",
                                "detail": "low"
                            }
                        })
                    if len(video_frames) > 0:
                        prompt_addition = f"\nNote: I've provided {len(video_frames)} keyframes from the video. "
                        user_content[0]["text"] = prompt + prompt_addition
                    else:
                        video_src = _get_video_url(video)
                        video_tensor = _get_video_tensor(video)
                        tensor_shape = getattr(video_tensor, "shape", None)
                        keys = list(video.keys())[:30] if isinstance(video, dict) else None
                        rn_pbar.error(f"视频解析失败：未能从 VIDEO 中获得可用帧（type={type(video)} src={video_src} tensor_type={type(video_tensor)} tensor_shape={tensor_shape} keys={keys}）")
                except Exception as e:
                    rn_pbar.error(f"视频处理出错: {str(e)}")

            messages = [
                {'role': 'system', 'content': f'{role}'},
                {'role': 'user', 'content': user_content},
            ]
        
        # 调用API
        try:
            completion = client.chat.completions.create(
                model=used_model, 
                messages=messages, 
                temperature=temperature
            )
            if completion is not None and hasattr(completion, 'choices') and len(completion.choices) > 0:
                prompt = completion.choices[0].message.content
            else:
                print(f"API Response invalid: {completion}")
                prompt = 'Error: No response from API (Empty choices)'
        except Exception as e:
            error_msg = str(e)
            if "not a multimodal model" in error_msg:
                prompt = f"{used_model}不是多模态模型，无法识别图片，请重新选择模型"
                rn_pbar.error(prompt)
                return (prompt,)

            print(f"API call error: {error_msg}")
            
            # 诊断：如果是 JSON 解析错误，尝试获取原始响应内容帮助调试
            if "Expecting value" in error_msg or "JSON" in error_msg:
                try:
                    import urllib.request
                    import urllib.error
                    
                    # 尝试构建完整 URL 进行测试
                    test_url = used_api_baseurl
                    if not test_url.endswith('/'):
                        test_url += '/'
                    test_url += 'chat/completions'
                    
                    print(f"Diagnosing API connection to: {test_url}")
                    
                    # 简单的测试请求头
                    headers = {
                        "Authorization": f"Bearer {used_api_key}",
                        "Content-Type": "application/json"
                    }
                    
                    # 最小化的测试 Payload
                    payload = json.dumps({
                        "model": used_model,
                        "messages": [{"role": "user", "content": "hi"}],
                        "max_tokens": 1
                    }).encode('utf-8')
                    
                    req = urllib.request.Request(test_url, data=payload, headers=headers, method='POST')
                    
                    try:
                        with urllib.request.urlopen(req, timeout=10) as response:
                            print(f"Diagnostic Response Code: {response.getcode()}")
                            body = response.read().decode('utf-8', errors='ignore')
                            print(f"Diagnostic Response Body (first 500 chars): {body[:500]}")
                    except urllib.error.HTTPError as he:
                        print(f"Diagnostic HTTP Error: {he.code}")
                        error_body = he.read().decode('utf-8', errors='ignore')
                        print(f"Diagnostic Error Body: {error_body[:500]}")
                        error_msg += f" (HTTP {he.code}: {error_body[:200]})"
                    except Exception as de:
                        print(f"Diagnostic check failed: {de}")
                        
                except Exception as diag_e:
                    print(f"Failed to run diagnostics: {diag_e}")

            prompt = f'Error: {error_msg}'
        if isinstance(prompt, str) and prompt.startswith('Error'):
            rn_pbar.error(prompt)
        else:
            rn_pbar.done(char_count=len(prompt or ""))
        return (prompt,)
