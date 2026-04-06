import os
import sys
import hashlib
import json
import argparse
import time
from datetime import datetime
from contextlib import asynccontextmanager
from openai import AsyncOpenAI
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import aiosqlite
import asyncio
import random
from confidence import answer_with_confidence, validate_answer

# 设置UTF-8编码以正确显示中文
os.environ['PYTHONIOENCODING'] = 'utf-8'
if sys.platform == 'win32':
    os.system('chcp 65001 >nul 2>&1')
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

load_dotenv()

ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")
CACHE_RETRY_PROBABILITY = float(os.getenv("CACHE_RETRY_PROBABILITY", "0.1"))

class LLMAnswerer:
    def __init__(self, api_key=None, model="gpt-3.5-turbo", db_path=None,
                 base_url=None, custom_headers=None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            raise ValueError(
                "未设置OPENAI_API_KEY。请在.env文件中设置或通过参数传入。\n"
                "请复制.env.example为.env并填入你的API密钥。"
            )

        self.model = model
        self.db_path = db_path or os.getenv("DB_PATH", "answer_cache.db")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.custom_headers = custom_headers or {}
        self.db_conn = None

        client_kwargs = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        if self.custom_headers:
            client_kwargs["default_headers"] = self.custom_headers

        self.client = AsyncOpenAI(**client_kwargs)

    async def connect_db(self):
        """建立数据库连接"""
        self.db_conn = await aiosqlite.connect(self.db_path)
        await self.db_conn.execute('PRAGMA journal_mode=WAL')
        await self.db_conn.execute('PRAGMA cache_size=-64000')
        await self.db_conn.execute('PRAGMA synchronous=NORMAL')
        await self.db_conn.commit()

    async def close_db(self):
        """关闭数据库连接"""
        if self.db_conn:
            await self.db_conn.close()
            self.db_conn = None

    async def init_database(self):
        """初始化SQLite数据库"""
        await self.db_conn.execute('''
            CREATE TABLE IF NOT EXISTS answer_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question_hash TEXT UNIQUE NOT NULL,
                title TEXT NOT NULL,
                options TEXT,
                question_type TEXT,
                answer TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        await self.db_conn.execute('CREATE INDEX IF NOT EXISTS idx_question_hash ON answer_cache(question_hash)')
        await self.db_conn.commit()

    def _get_cache_key(self, title, options):
        """生成缓存键"""
        content = f"{title}|{options or ''}"
        return hashlib.md5(content.encode()).hexdigest()

    async def _get_cached_answer(self, cache_key):
        """从数据库获取缓存答案"""
        cursor = await self.db_conn.execute('SELECT answer FROM answer_cache WHERE question_hash = ?', (cache_key,))
        result = await cursor.fetchone()
        return result[0] if result else None

    async def _save_to_cache(self, cache_key, title, options, question_type, answer):
        """保存答案到数据库"""
        try:
            await self.db_conn.execute('''
                INSERT OR REPLACE INTO answer_cache
                (question_hash, title, options, question_type, answer)
                VALUES (?, ?, ?, ?, ?)
            ''', (cache_key, title, options, question_type, answer))
            await self.db_conn.commit()
        except Exception as e:
            print(f"保存缓存失败: {e}")

    async def _call_llm(self, title, options=None, question_type=None):
        """调用OpenAI API（带置信度判断版本）"""
        answer = await answer_with_confidence(
            client=self.client,
            model=self.model,
            title=title,
            options=options,
            question_type=question_type
        )
        return answer

    async def answer_question(self, title, options=None, question_type=None, skip_cache=False):
        """
        将题目转换为LLM请求并获取答案
        返回格式: [error_msg, answer, elapsed_time]
        """
        start_time = time.time()
        cache_key = self._get_cache_key(title, options)

        if not skip_cache:
            cached_answer = await self._get_cached_answer(cache_key)
            if cached_answer:
                elapsed = time.time() - start_time
                if random.random() < CACHE_RETRY_PROBABILITY:
                    print(f"[缓存命中-随机重试] 题目: {title[:50]}... -> 旧答案: {cached_answer} (耗时: {elapsed*1000:.0f}ms)")
                else:
                    print(f"[缓存命中] 题目: {title[:50]}... -> 答案: {cached_answer} (耗时: {elapsed*1000:.0f}ms)")
                    return [None, cached_answer, elapsed]

        # confidence.py 已经实现了重试和验证机制，这里只需要调用一次
        try:
            answer = await self._call_llm(title, options, question_type)
            elapsed = time.time() - start_time

            # confidence.py 内部已经做了验证，但这里再次验证以确保万无一失
            if validate_answer(answer, question_type):
                await self._save_to_cache(cache_key, title, options, question_type, answer)
                print(f"[LLM回答] 题目: {title[:50]}... -> 答案: {answer} (耗时: {elapsed*1000:.0f}ms)")
                return [None, answer, elapsed]
            else:
                # 理论上不应该到这里，因为 confidence.py 已经验证过
                elapsed = time.time() - start_time
                print(f"[警告] confidence.py 返回了无效答案: {answer} (耗时: {elapsed*1000:.0f}ms)")
                return ["LLM返回的答案格式不规范", None, elapsed]

        except Exception as e:
            elapsed = time.time() - start_time
            print(f"[请求失败] {str(e)} (耗时: {elapsed*1000:.0f}ms)")
            return [f"LLM请求失败: {str(e)}", None, elapsed]

    def get_config_info(self):
        """获取配置信息"""
        return {
            "model": self.model,
            "base_url": self.base_url or "https://api.openai.com/v1",
            "db_path": self.db_path,
            "api_key_set": bool(self.api_key)
        }

GLOBAL_SKIP_CACHE = False
answerer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    await answerer.connect_db()
    await answerer.init_database()
    yield
    await answerer.close_db()

app = FastAPI(lifespan=lifespan)

def print_startup_info(answerer_obj, port):
    """打印启动信息"""
    config = answerer_obj.get_config_info()

    # 获取额外的配置信息
    exa_api_key = os.getenv("EXA_API_KEY")
    confidence_threshold = float(os.getenv("CONFIDENCE_THRESHOLD", "0.7"))

    print("\n" + "="*60)
    print("LLM智能答题服务启动成功（异步版本）")
    print("="*60)
    print(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"服务地址: http://localhost:{port}")
    print(f"API端点: http://localhost:{port}/search")

    # 功能状态摘要
    print("-"*60)
    print("启用功能:")
    features = []
    features.append(f"  ✓ 智能缓存 (随机重试概率: {CACHE_RETRY_PROBABILITY*100:.0f}%)")
    if exa_api_key:
        features.append(f"  ✓ 置信度评估 + 联网搜索 (阈值: {confidence_threshold:.1f})")
    else:
        features.append(f"  ✓ 置信度评估 (阈值: {confidence_threshold:.1f})")
        features.append(f"  ✗ 联网搜索 (未配置 EXA_API_KEY)")
    if ACCESS_TOKEN:
        features.append(f"  ✓ 访问令牌认证")
    else:
        features.append(f"  ✗ 访问令牌认证 (未启用)")

    for feature in features:
        print(feature)

    # LLM配置
    print("-"*60)
    print("LLM配置:")
    print(f"  模型: {config['model']}")
    print(f"  API地址: {config['base_url']}")
    print(f"  API密钥: {'已设置' if config['api_key_set'] else '未设置'}")

    # 存储配置
    print("-"*60)
    print("存储配置:")
    print(f"  数据库: {config['db_path']}")
    print(f"  缓存策略: MD5哈希 + 随机重试")

    # Exa搜索配置
    if exa_api_key:
        print("-"*60)
        print("联网搜索配置:")
        print(f"  Exa API: 已配置")
        print(f"  搜索触发: 置信度 < {confidence_threshold:.1f}")

    # AnswererWrapper配置
    print("-"*60)
    print("AnswererWrapper配置:")
    print("[")

    headers_config = {"Content-Type": "application/json"}
    if ACCESS_TOKEN:
        headers_config["X-Access-Token"] = ACCESS_TOKEN

    print(json.dumps({
        "name": "LLM智能答题",
        "url": f"http://localhost:{port}/search",
        "method": "post",
        "contentType": "json",
        "type": "GM_xmlhttpRequest",
        "headers": headers_config,
        "data": {
            "title": "${title}",
            "options": "${options}",
            "type": "${type}"
        },
        "handler": "return (res) => res.code === 1 ? [undefined, res.answer] : [res.msg, undefined]"
    }, ensure_ascii=False, indent=2))
    print("]")
    print("="*60 + "\n")

@app.get('/')
@app.head('/')
async def heartbeat():
    """心跳检查接口"""
    return "服务已启动"

@app.get('/search')
@app.post('/search')
async def search(request: Request):
    """模拟题库API接口，实际使用LLM生成答案"""
    if request.method == 'GET':
        params = dict(request.query_params)
        title = params.get('title', '')
        options = params.get('options')
        question_type = params.get('type')
        skip_cache = GLOBAL_SKIP_CACHE or params.get('skip_cache', 'false').lower() == 'true'
        token = request.headers.get('X-Access-Token') or params.get('token')

        if title and sys.platform == 'win32':
            try:
                title = title.encode('latin1').decode('utf-8')
            except (UnicodeDecodeError, UnicodeEncodeError):
                pass

        if options and sys.platform == 'win32':
            try:
                options = options.encode('latin1').decode('utf-8')
            except (UnicodeDecodeError, UnicodeEncodeError):
                pass
    else:
        data = await request.json()
        title = data.get('title', '')
        options = data.get('options')
        question_type = data.get('type')
        skip_cache = GLOBAL_SKIP_CACHE or data.get('skip_cache', False)
        token = request.headers.get('X-Access-Token') or request.query_params.get('token') or data.get('token')

    if ACCESS_TOKEN and token != ACCESS_TOKEN:
        return JSONResponse({"code": 0, "msg": "无效的访问令牌"}, status_code=401)

    print(f"\n[收到请求] {datetime.now().strftime('%H:%M:%S')} - 题型: {question_type or '未知'}")
    print(f"  题目: {title[:100]}{'...' if len(title) > 100 else ''}")
    if options:
        print(f"  选项: {options[:100]}{'...' if len(options) > 100 else ''}")
    if skip_cache:
        print(f"  跳过缓存: 是")

    if not title:
        return JSONResponse({"code": 0, "msg": "题目不能为空"})

    result = await answerer.answer_question(title, options, question_type, skip_cache)
    error_msg, answer, elapsed_time = result if len(result) == 3 else (*result, 0)

    if answer:
        response_data = {
            "code": 1,
            "question": title,
            "answer": answer
#            ,"elapsed_time": round(elapsed_time, 3),
#            "elapsed_ms": round(elapsed_time * 1000, 0)
        }
        print(f"[响应成功] 答案: {answer}, 总耗时: {elapsed_time*1000:.0f}ms")
        return JSONResponse(response_data)
    else:
        response_data = {
            "code": 0,
            "msg": error_msg or "未知错误"
#            ,"elapsed_time": round(elapsed_time, 3),
#            "elapsed_ms": round(elapsed_time * 1000, 0)
        }
        print(f"[响应失败] 错误: {error_msg}, 总耗时: {elapsed_time*1000:.0f}ms")
        return JSONResponse(response_data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LLM智能答题服务')
    parser.add_argument('-skipcache', '--skip-cache', action='store_true',
                        help='跳过缓存，所有请求直接调用LLM API')
    args = parser.parse_args()

    GLOBAL_SKIP_CACHE = args.skip_cache

    port = int(os.getenv("LISTEN_PORT", 5000))
    model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    base_url = os.getenv("OPENAI_BASE_URL")

    custom_headers = {
        "User-Agent": "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "X-Client-Name": "question-libraries"
    }
    answerer = LLMAnswerer(model=model, base_url=base_url, custom_headers=custom_headers)

    print_startup_info(answerer, port)

    if GLOBAL_SKIP_CACHE:
        print("⚠️  缓存已全局禁用 - 所有请求将直接调用LLM API")
        print("="*60 + "\n")

    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=port)
