from openai import OpenAI
import time
import json
import os
import re
from PIL import Image
import numpy as np
import base64



def get_config():
    try:
        # 修改配置文件路径为新的配置文件
        config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config",
                                  'ComfyUI_rn_translator-config.json')
        
        with open(config_path, 'r', encoding='utf-8') as f:  
            config = json.load(f)
        
        # 从llm部分获取当前provider的配置
        llm_config = config.get('llm', {})
        current_provider = llm_config.get('current_provider', 'quchi')
        providers = llm_config.get('providers', {})
        
        # 获取当前provider的配置
        if current_provider in providers:
            provider_config = providers[current_provider]
            # 返回包含api_key的字典，保持与原函数兼容
            return {
                'api_key': provider_config.get('api_key', ''),
                'model': provider_config.get('model', ''),
                'base_url': provider_config.get('base_url', ''),
                'temperature': provider_config.get('temperature', 0.7),
                'max_tokens': provider_config.get('max_tokens', 1000),
                'top_p': provider_config.get('top_p', 0.9)
            }
        else:
            # 如果当前provider不存在，返回空字典
            return {}
            
    except Exception as e:
        print(f"Error loading config: {e}")
        return {}


def save_config(config):
    try:
        # 修改配置文件路径为新的配置文件
        config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config",
                                  'ComfyUI_RN_External_Interface-config.json')
        
        # 先读取现有的配置
        with open(config_path, 'r', encoding='utf-8') as f:
            existing_config = json.load(f)
        
        # 从llm部分获取当前provider
        llm_config = existing_config.get('llm', {})
        current_provider = llm_config.get('current_provider', 'deepseek')
        providers = llm_config.get('providers', {})
        
        # 更新当前provider的api_key
        if current_provider in providers:
            # 只更新api_key，保持其他配置不变
            if 'api_key' in config:
                providers[current_provider]['api_key'] = config['api_key']
            
            # 可选：更新其他字段
            for key in ['model', 'base_url', 'temperature', 'max_tokens', 'top_p']:
                if key in config:
                    providers[current_provider][key] = config[key]
            
            # 更新回配置
            existing_config['llm']['providers'] = providers
            
            # 写回文件
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(existing_config, f, indent=2, ensure_ascii=False)
                
    except Exception as e:
        print(f"Error saving config: {e}")


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
            
        # 确保值在0-255范围
        if frame_np.max() <= 1.0:
            frame_np = frame_np * 255.0
        
        # 转换为uint8
        frame_img = Image.fromarray(np.clip(frame_np, 0, 255).astype(np.uint8))
        
        # 调整大小（与图片处理保持一致）
        lsize = np.max(frame_img.size)
        factor = 1
        while lsize / factor > 1024:  # 视频帧可以稍小一些
            factor *= 2
        if factor > 1:
            frame_img = frame_img.resize((frame_img.size[0] // factor, frame_img.size[1] // factor))
        
        # 保存为临时文件并编码
        frame_path = f'frame_{time.time()}_{idx}.webp'
        frame_img.save(frame_path, 'WEBP')
        
        with open(frame_path, "rb") as f:
            frame_base64 = base64.b64encode(f.read()).decode('utf-8')
        
        frames.append(frame_base64)
        os.remove(frame_path)
    
    return frames


class RN_Translator():
    def __init__(self):
        pass

    def _load_llm_config(self):
        cfg_path = os.path.join(os.path.dirname(__file__), "config", "comfyui_rn_translator-config.json")
        if not os.path.exists(cfg_path):
            return {}
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            llm = data.get("llm") or {}
            current = llm.get("current_provider")
            providers = llm.get("providers") or {}
            provider_cfg = providers.get(current) or {}
            return provider_cfg
        except Exception:
            return {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "prompt"}),
                "advanced_options": (["auto", "force_english", "force_chinese"], {"default": "auto"}),
            },
            "optional": {
                "seed": ("INT", {"default": 100, "min": 0, "max": 0xffffffffffffffff}),
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
        if apiBaseUrl == "default":
            apiBaseUrl = ""
        if apiKey == "default":
            apiKey = ""
        if model == "default":
            model = ""

        cfg = get_config()
        cfg_base_url = cfg.get("base_url")
        cfg_api_key = cfg.get("api_key")
        cfg_model = cfg.get("model")
        cfg_temperature = cfg.get("temperature")
        cfg_max_tokens = cfg.get("max_tokens")
        cfg_top_p = cfg.get("top_p")

        used_api_baseurl = (apiBaseUrl or cfg_base_url or "https://ai.t8star.cn//v1")
        used_model = (model or cfg_model or "gpt-4o-mini")
        used_api_key = (apiKey or cfg_api_key or "")
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
            use_temperature = (
                temperature if temperature is not None else (
                    cfg_temperature if cfg_temperature is not None else 0.3
                )
            )
            use_max_tokens = cfg_max_tokens if cfg_max_tokens is not None else 512
            params = {
                "model": used_model,
                "messages": messages,
                "temperature": use_temperature,
                "max_tokens": use_max_tokens,
            }
            if cfg_top_p is not None:
                params["top_p"] = cfg_top_p
            completion = client.chat.completions.create(**params)
            if completion is not None and hasattr(completion, 'choices') and len(completion.choices) > 0:
                return completion.choices[0].message.content.strip()
            return "错误：API返回空结果"
        except Exception as e:
            return f"翻译错误：{str(e)}"

    def translate_text(self, prompt, advanced_options, seed=0):
        cleaned = re.sub(r'[\x00\x01-\x08\x0b\x0c\x0e-\x1f\x7f]', '', prompt or "")
        if not cleaned.strip():
            return ("错误：请输入要翻译的文本",)
        src_label, dst_label = self._detect_direction(cleaned, advanced_options)
        chunks = self._split_text_into_chunks(cleaned, max_chunk_size=400)
        if len(chunks) == 1:
            res = self._translate_chunk(chunks[0], src_label, dst_label)
            return (res,)
        translated = []
        for c in chunks:
            translated.append(self._translate_chunk(c, src_label, dst_label))
        return (' '.join(translated),)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float(time.time())


class RN_Prompt_Translator():
    def __init__(self):
        pass

    def _load_llm_config(self):
        cfg_path = os.path.join(os.path.dirname(__file__), "config", "comfyui_rn_translator-config.json")
        if not os.path.exists(cfg_path):
            return {}
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            llm = data.get("llm") or {}
            current = llm.get("current_provider")
            providers = llm.get("providers") or {}
            provider_cfg = providers.get(current) or {}
            return provider_cfg
        except Exception:
            return {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "advanced_options": (["auto", "force_English", "force_Chinese"], {"default": "auto"}),
            },
            "optional": {
                "seed": ("INT", {"default": 100, "min": 0, "max": 0xffffffffffffffff}),
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
        if apiBaseUrl == "default":
            apiBaseUrl = ""
        if apiKey == "default":
            apiKey = ""
        if model == "default":
            model = ""

        cfg = get_config()
        cfg_base_url = cfg.get("base_url")
        cfg_api_key = cfg.get("api_key")
        cfg_model = cfg.get("model")
        cfg_temperature = cfg.get("temperature")
        cfg_max_tokens = cfg.get("max_tokens")
        cfg_top_p = cfg.get("top_p")

        used_api_baseurl = (apiBaseUrl or cfg_base_url or "https://ai.t8star.cn//v1")
        used_model = (model or cfg_model or "gpt-4o-mini")
        used_api_key = (apiKey or cfg_api_key or "")
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
            use_temperature = (
                temperature if temperature is not None else (
                    cfg_temperature if cfg_temperature is not None else 0.3
                )
            )
            use_max_tokens = cfg_max_tokens if cfg_max_tokens is not None else 512
            params = {
                "model": used_model,
                "messages": messages,
                "temperature": use_temperature,
                "max_tokens": use_max_tokens,
            }
            if cfg_top_p is not None:
                params["top_p"] = cfg_top_p
            completion = client.chat.completions.create(**params)
            if completion is not None and hasattr(completion, 'choices') and len(completion.choices) > 0:
                return completion.choices[0].message.content.strip()
            return "错误：API返回空结果"
        except Exception as e:
            return f"翻译错误：{str(e)}"

    def translate_text(self, prompt, advanced_options, seed=0):
        cleaned = re.sub(r'[\x00\x01-\x08\x0b\x0c\x0e-\x1f\x7f]', '', prompt or "")
        if not cleaned.strip():
            return ("错误：请输入要翻译的文本",)
        src_label, dst_label = self._detect_direction(cleaned, advanced_options)
        chunks = self._split_text_into_chunks(cleaned, max_chunk_size=400)
        if len(chunks) == 1:
            res = self._translate_chunk(chunks[0], src_label, dst_label)
            return (res,)
        translated = []
        for c in chunks:
            translated.append(self._translate_chunk(c, src_label, dst_label))
        return (' '.join(translated),)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float(time.time())


class RN_Midjourney_Prompter():
    def __init__(self):
        pass

    def _load_llm_config(self):
        cfg_path = os.path.join(os.path.dirname(__file__), "config", "comfyui_rn_translator-config.json")
        if not os.path.exists(cfg_path):
            return {}
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            llm = data.get("llm") or {}
            current = llm.get("current_provider")
            providers = llm.get("providers") or {}
            provider_cfg = providers.get(current) or {}
            return provider_cfg
        except Exception:
            return {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "seed": ("INT", {"default": 100, "min": 0, "max": 0xffffffffffffffff}),
                "temperature": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 2.0, "step": 0.1}),
                # "apiBaseUrl": ("STRING", {"default": "default"}),
                # "apiKey": ("STRING", {"default": "default"}),
                # "model": ("STRING", {"default": "default"}),
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
        if apiBaseUrl == "default":
            apiBaseUrl = ""
        if apiKey == "default":
            apiKey = ""
        if model == "default":
            model = ""

        cfg = get_config()
        cfg_base_url = cfg.get("base_url")
        cfg_api_key = cfg.get("api_key")
        cfg_model = cfg.get("model")
        cfg_temperature = cfg.get("temperature")
        cfg_max_tokens = cfg.get("max_tokens")
        cfg_top_p = cfg.get("top_p")

        used_api_baseurl = (apiBaseUrl or cfg_base_url or "https://ai.t8star.cn//v1")
        used_model = (model or cfg_model or "gpt-4o-mini")
        used_api_key = (apiKey or cfg_api_key or "")
        if not used_api_key:
            return "错误：请提供API密钥"

        try:
            client = OpenAI(api_key=used_api_key, base_url=used_api_baseurl)
            system_prompt = "你是资深提示词工程师，专注将输入内容重写为 Midjourney 风格英文提示词。"
            user_prompt = f"""
                            Rewrite the following content as a Midjourney-style prompt in English only.
                            Requirements:
                            - concise, comma-separated tags
                            - include subject, medium, style, lighting, composition, color palette
                            - add camera/lens and mood if relevant
                            - no parameters/flags (e.g., --ar, --v, --s)
                            - no explanations or extra text, return only the final prompt
                            Content:
                            {chunk}
                            """
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt.strip()},
            ]
            use_temperature = (
                temperature if temperature is not None else (
                    cfg_temperature if cfg_temperature is not None else 0.3
                )
            )
            use_max_tokens = cfg_max_tokens if cfg_max_tokens is not None else 512
            params = {
                "model": used_model,
                "messages": messages,
                "temperature": use_temperature,
                "max_tokens": use_max_tokens,
            }
            if cfg_top_p is not None:
                params["top_p"] = cfg_top_p
            completion = client.chat.completions.create(**params)
            if completion is not None and hasattr(completion, 'choices') and len(completion.choices) > 0:
                return completion.choices[0].message.content.strip()
            return "错误：API返回空结果"
        except Exception as e:
            return f"生成错误：{str(e)}"

    def generate_mj_prompt(self, prompt, seed=0, temperature=0.3, apiBaseUrl="default", apiKey="default", model="default"):
        cleaned = re.sub(r'[\x00\x01-\x08\x0b\x0c\x0e-\x1f\x7f]', '', prompt or "")
        if not cleaned.strip():
            return ("错误：请输入要生成的文本",)
        chunks = self._split_text_into_chunks(cleaned, max_chunk_size=400)
        if len(chunks) == 1:
            res = self._generate_chunk_mj(chunks[0], temperature=temperature, apiBaseUrl=apiBaseUrl, apiKey=apiKey, model=model)
            return (res,)
        generated = []
        for c in chunks:
            generated.append(self._generate_chunk_mj(c, temperature=temperature, apiBaseUrl=apiBaseUrl, apiKey=apiKey, model=model))
        return (' '.join(generated),)

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
        cfg = get_config()
        used_api_baseurl = (api_baseurl or cfg.get("base_url"))
        used_api_key = (api_key or cfg.get("api_key") or "")
        used_model = (model or cfg.get("model") or "gpt-4o-mini")
        
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
                # 提取视频帧
                video_frames = extract_video_frames(video, max_frames=max_video_frames)
                
                # 为每个帧添加图像内容
                for i, frame_base64 in enumerate(video_frames):
                    user_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/webp;base64,{frame_base64}",
                            "detail": "medium"
                        }
                    })
                
                # 如果需要，可以在prompt中说明这些是视频帧
                if len(video_frames) > 1:
                    prompt_addition = f"\nNote: I've provided {len(video_frames)} keyframes from the video. "
                    user_content[0]["text"] = prompt + prompt_addition
                    
            except Exception as e:
                print(f"视频处理出错: {str(e)}")
                # 可以选择返回错误信息或继续处理
        
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
            
        return (response,)


class RN_LLMAPI_Pro_Node():

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (["doubao-seed-1-6-251015", 
                "DeepSeek-V3",
                "Qwen3-235B-A22B-Instruct-2507", 
                "gemini-2.5-flash", 
                "gpt-5",
                "gemini-3-pro-preview"], {"default": "doubao-seed-1-6-251015"}),
                "role": ("STRING", {"multiline": True, "default": "You are a helpful assistant"}),
                "prompt": ("STRING", {"multiline": True, "default": "Hello"}),
                "temperature": ("FLOAT", {"default": 0.6}),
                "seed": ("INT", {"default": -1}),
            },
            "optional": {
                "ref_image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "rn_run_llmapi_pro"
    CATEGORY = "RunNode/rn_prompter"

    def rn_run_llmapi_pro(self, api_baseurl, api_key, model, role, prompt, temperature, seed, ref_image=None):
        cfg = get_config()
        # used_api_baseurl = (api_baseurl or env_api_baseurl or cfg.get("base_url") or "https://ai.t8star.cn//v1")
        used_api_baseurl = (api_baseurl or cfg.get("base_url"))
        used_api_key = (api_key or cfg.get("api_key") or "")
        used_model = (model or cfg.get("model") or "doubao-seed-1-6-251015")
        client = OpenAI(api_key=used_api_key, base_url=used_api_baseurl)
        if ref_image is None:
            messages = [
                {'role': 'system', 'content': f'{role}'},
                {'role': 'user', 'content': f'{prompt}'},
            ]
        else:
            base64_image = encode_image_b64(ref_image)
            messages = [
                {'role': 'system', 'content': f'{role}'},
                {'role': 'user', 
                 'content': [
                        {
                            "type": "text",
                            "text": f"{prompt}"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        },
                    ]},
            ]
        completion = client.chat.completions.create(model=used_model, messages=messages, temperature=temperature)
        if completion is not None and hasattr(completion, 'choices'):
            prompt = completion.choices[0].message.content
        else:
            prompt = 'Error'
        return (prompt,)
