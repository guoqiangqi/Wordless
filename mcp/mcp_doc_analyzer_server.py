#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能文档分析器 MCP 服务器

基于 Wordless NLP 库实现的文档分析服务，通过 Model Context Protocol (MCP) 提供：
- 可读性分析（ARI、Lix、Coleman-Liau Index）
- 词汇多样性分析（TTR、RTTR、CTTR、Herdan's C、Yule's K）
- 结构复杂度分析（句长、词长统计）
- 智能语言检测（中英文，简繁体，美英式）

支持的MCP功能：
- Tools: 2个工具（analyze_document, detect_language）
- Resources: 2个资源（语言列表、指标说明）
- Prompts: 3个提示模板（分析、对比、改进）

使用方法:
    stdio模式:  python mcp_doc_analyzer_server.py
    HTTP模式:   python mcp_doc_analyzer_server.py --transport streamable-http --port 8000

参考文档: https://github.com/modelcontextprotocol/python-sdk
"""

import argparse
import json
import logging
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np

from mcp.server.fastmcp import FastMCP
from mcp.server.streamable_http import TransportSecuritySettings

# 添加Wordless模块到路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 延迟导入Wordless模块（仅在实际使用时导入）
wl_sentence_tokenization = None
wl_word_tokenization = None
wl_settings_global = None

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BearerTokenMiddleware:
    """简单的Bearer Token认证中间件"""
    
    def __init__(self, app, valid_token: str):
        """
        Args:
            app: ASGI应用
            valid_token: 有效的Bearer token
        """
        self.app = app
        self.valid_token = valid_token
        logger.info("✅ Bearer Token认证已启用")
    
    async def __call__(self, scope, receive, send):
        """ASGI调用"""
        if scope["type"] == "http":
            # 检查Authorization头
            headers = dict(scope.get("headers", []))
            auth_header = headers.get(b"authorization", b"").decode()
            
            if auth_header.startswith("Bearer "):
                token = auth_header[7:]  # 去掉 "Bearer "前缀
                if token == self.valid_token:
                    # 认证成功，继续处理请求
                    await self.app(scope, receive, send)
                    return
            
            # 认证失败，返回401
            await send({
                "type": "http.response.start",
                "status": 401,
                "headers": [[b"content-type", b"application/json"]],
            })
            await send({
                "type": "http.response.body",
                "body": b'{"error": "Unauthorized", "message": "Invalid or missing Bearer token"}',
            })
            return
        
        # 非HTTP请求（如WebSocket），直接传递
        await self.app(scope, receive, send)


def _import_wordless_modules():
    """延迟导入Wordless模块"""
    global wl_sentence_tokenization, wl_word_tokenization, wl_settings_global
    
    if wl_sentence_tokenization is None:
        try:
            from wordless.wl_nlp import (
                wl_sentence_tokenization as _st,
                wl_word_tokenization as _wt,
            )
            from wordless.wl_settings import wl_settings_global as _wsg
            
            wl_sentence_tokenization = _st
            wl_word_tokenization = _wt
            wl_settings_global = _wsg
        except ImportError as e:
            logger.error(f"无法导入Wordless模块: {e}")
            logger.error("请确保已安装所有依赖：pip install -r requirements.txt")
            raise


class MockMain:
    """模拟Wordless主窗口对象，提供NLP配置"""
    
    def __init__(self):
        """初始化配置"""
        _import_wordless_modules()
        self.settings_global = wl_settings_global.init_settings_global()
        self.settings_custom = {
            'sentence_tokenization': {
                'sentence_tokenizer_settings': {
                    'zho_cn': 'spacy_dependency_parser_zho',
                    'zho_tw': 'spacy_dependency_parser_zho',
                    'eng_us': 'spacy_dependency_parser_eng',
                    'eng_gb': 'spacy_dependency_parser_eng',
                    'other': 'spacy_dependency_parser_eng'
                }
            },
            'word_tokenization': {
                'word_tokenizer_settings': {
                    'zho_cn': 'spacy_zho',
                    'zho_tw': 'spacy_zho',
                    'eng_us': 'spacy_eng',
                    'eng_gb': 'spacy_eng',
                    'other': 'spacy_eng'
                }
            },
            'files': {
                'misc_settings': {
                    'read_files_in_chunks_lines': 1000
                }
            },
            'measures': {
                'lexical_density_diversity': {
                    'hdd': {'sample_size': 42},
                    'msttr': {'num_tokens_in_each_seg': 100},
                    'mtld': {'factor_size': 0.72},
                    'mattr': {'window_size': 100}
                }
            }
        }


class LanguageDetector:
    """智能语言检测器（优化版）"""
    
    # 扩展简体中文高频字（前50个最常用字）
    SIMPLIFIED_CHARS = '的一是了我不人在他有这个上们来到时大地为子中你说生国年着就那和要她出也得里后自以会家可下而过天去能对小么心多之起成好看见只名没用主金开手知道些日四正当想行理分走见实西面山实明王美情百题海'
    
    # 扩展繁体中文特征字（台湾常用繁体字）
    TRADITIONAL_CHARS = '個們說處時間話頭東標題開關係條幾種學過電錢買賣實際認識讓變動產發現體業會員專業資訊網絡連線應該問題國際'
    
    # 扩展英式拼写词汇
    BRITISH_PATTERNS = [
        'colour', 'favour', 'honour', 'labour', 'neighbour', 'rumour', 'savour',
        'centre', 'theatre', 'metre', 'litre', 'fibre',
        'realise', 'organise', 'recognise', 'analyse', 'summarise',
        'defence', 'offence', 'licence', 'practise',
        'travelled', 'cancelled', 'modelling', 'labelled'
    ]
    
    # 扩展美式拼写词汇
    AMERICAN_PATTERNS = [
        'color', 'favor', 'honor', 'labor', 'neighbor', 'rumor', 'savor',
        'center', 'theater', 'meter', 'liter', 'fiber',
        'realize', 'organize', 'recognize', 'analyze', 'summarize',
        'defense', 'offense', 'license', 'practice',
        'traveled', 'canceled', 'modeling', 'labeled'
    ]
    
    @staticmethod
    def detect_language(text: str) -> Optional[str]:
        """
        自动检测文本语言（优化版V2：更准确的技术文档识别）
        
        改进要点：
        1. 更全面的代码清理（变量名、命令参数、整行命令等）
        2. 只统计有意义字符（排除标点、数字、符号）
        3. 使用相对比例判断（中文vs英文）而非绝对阈值
        4. 扩展简繁体特征字库（100+字符）
        5. 扩展英美式词汇库（24个词汇对）
        6. 智能识别命令行和代码行并移除
        
        返回: 'zho_cn', 'zho_tw', 'eng_us', 'eng_gb' 或 None
        """
        if not text or len(text) == 0:
            return None
        
        # === 阶段1: 预处理清理 ===
        text_cleaned = text
        
        # 移除Markdown代码块
        text_cleaned = re.sub(r'```[\s\S]*?```', ' ', text_cleaned)
        text_cleaned = re.sub(r'`[^`]+`', ' ', text_cleaned)
        
        # 移除整行shell命令（以常见命令开头的行）
        command_patterns = r'^\s*(?:docker|bash|sh|python|pip|npm|yarn|git|cd|ls|mkdir|rm|cp|mv|cat|echo|export|source|chmod|chown|wget|curl|make|cmake|gcc|g\+\+|apt|yum|brew|sudo|npu-smi|msprof)[\s\w\-\./]*$'
        text_cleaned = re.sub(command_patterns, ' ', text_cleaned, flags=re.MULTILINE)
        
        # 移除URL和邮箱
        text_cleaned = re.sub(r'https?://[^\s\u4e00-\u9fff]+', ' ', text_cleaned)
        text_cleaned = re.sub(r'\S+@\S+\.\S+', ' ', text_cleaned)
        
        # 移除文件路径和扩展名
        text_cleaned = re.sub(r'[/\\][\w\-/\\]+\.[\w]+', ' ', text_cleaned)
        text_cleaned = re.sub(r'\.(?:sh|py|cpp|h|hpp|c|js|ts|json|xml|yaml|yml|md|txt|log|conf|ini|so|lib|dll|exe|run)\b', ' ', text_cleaned)
        
        # 移除命令行参数和环境变量
        text_cleaned = re.sub(r'--[\w\-]+=?[\w\-]*', ' ', text_cleaned)
        text_cleaned = re.sub(r'-[\w](?:\s|$)', ' ', text_cleaned)
        text_cleaned = re.sub(r'\$\{?[\w_]+\}?', ' ', text_cleaned)
        text_cleaned = re.sub(r'%[\w_]+%', ' ', text_cleaned)  # Windows环境变量
        
        # 移除版本号和IP地址
        text_cleaned = re.sub(r'\d+\.\d+\.\d+\.\d+', ' ', text_cleaned)
        text_cleaned = re.sub(r'\bv?\d+\.\d+\.\d+[\w\-\.]*', ' ', text_cleaned)
        
        # 移除变量名模式（snake_case, camelCase, CONSTANT_CASE等）
        text_cleaned = re.sub(r'\b[a-z]+_[a-z_0-9]+\b', ' ', text_cleaned, flags=re.IGNORECASE)
        text_cleaned = re.sub(r'\b[a-z]+[A-Z][a-zA-Z0-9]*\b', ' ', text_cleaned)
        text_cleaned = re.sub(r'\b[A-Z_]{2,}\b', ' ', text_cleaned)  # 常量名
        
        # === 阶段2: 统计有意义字符 ===
        chinese_chars = re.findall(r'[\u4e00-\u9fff]', text_cleaned)
        english_chars = re.findall(r'[a-zA-Z]', text_cleaned)
        
        chinese_count_cleaned = len(chinese_chars)
        english_count_cleaned = len(english_chars)
        meaningful_cleaned = chinese_count_cleaned + english_count_cleaned
        
        # 智能回退：只有在清理后几乎没有有效字符时才回退
        # 关键优化：如果清理后有中文内容（即使很少），保留清理结果
        if meaningful_cleaned < 5 or (meaningful_cleaned < 10 and chinese_count_cleaned == 0):
            # 回退条件：1) 总数<5字符，或 2) <10字符且没有中文
            text_cleaned = text
            chinese_chars = re.findall(r'[\u4e00-\u9fff]', text_cleaned)
            english_chars = re.findall(r'[a-zA-Z]', text_cleaned)
        
        chinese_count = len(chinese_chars)
        english_count = len(english_chars)
        meaningful_total = chinese_count + english_count
        
        # 如果有意义字符太少，无法判断
        if meaningful_total == 0:
            return None
        
        # === 阶段3: 计算相对比例 ===
        chinese_ratio = chinese_count / meaningful_total
        english_ratio = english_count / meaningful_total
        
        # === 阶段3.5: 前导语言检测（针对技术文档优化） ===
        # 检测文本开头的语言倾向（前20个有意义字符）
        text_start = text_cleaned[:100]  # 取前100个字符
        start_chinese = len(re.findall(r'[\u4e00-\u9fff]', text_start))
        start_english = len(re.findall(r'[a-zA-Z]', text_start))
        
        # 如果开头有明显中文（>=3个且比英文多），增加中文倾向
        has_chinese_leading = (start_chinese >= 3 and start_chinese >= start_english)
        
        # === 阶段4: 语言判断（多层次智能策略） ===
        
        # 特殊处理1：极短文本（<10个有意义字符）
        if meaningful_total < 10:
            # 优先按绝对数量判断
            if chinese_count >= 2:
                return 'zho_cn'
            elif english_count >= 4:
                return 'eng_us'
            elif chinese_count > 0:
                return 'zho_cn'
            elif english_count > 0:
                return 'eng_us'
            else:
                return None
        
        # 特殊处理2：短文本（10-25字符）- 使用更低的阈值
        elif meaningful_total <= 25:
            # 前导中文判断
            if has_chinese_leading and chinese_count >= 3:
                return 'zho_cn'
            # 中文信息密度高，占比>=20%就判定为中文
            elif chinese_count > 0 and chinese_ratio >= 0.2:
                return 'zho_cn'
            elif chinese_count > english_count:
                return 'zho_cn'
            elif english_count > 0:
                return 'eng_us'
            else:
                return None
        
        # 长文本（>=20字符）使用复杂策略
        
        # 策略0: 前导中文+足够中文数量
        if has_chinese_leading and chinese_count >= 5 and chinese_ratio >= 0.15:
            # 简繁体区分
            simplified_count = sum(1 for char in chinese_chars if char in LanguageDetector.SIMPLIFIED_CHARS)
            traditional_count = sum(1 for char in chinese_chars if char in LanguageDetector.TRADITIONAL_CHARS)
            
            if traditional_count >= 5 and traditional_count > simplified_count * 1.3:
                return 'zho_tw'
            else:
                return 'zho_cn'
        
        # 策略1: 中文明显占优（>30%）或中文略占优且绝对数量足够
        elif chinese_ratio > 0.3 or (chinese_ratio > 0.15 and chinese_count >= 20):
            # 简繁体区分
            simplified_count = sum(1 for char in chinese_chars if char in LanguageDetector.SIMPLIFIED_CHARS)
            traditional_count = sum(1 for char in chinese_chars if char in LanguageDetector.TRADITIONAL_CHARS)
            
            # 如果繁体特征明显（至少5个繁体字，且明显多于简体）
            if traditional_count >= 5 and traditional_count > simplified_count * 1.3:
                return 'zho_tw'
            else:
                return 'zho_cn'
        
        # 策略2: 英文明显占优（>60%）
        elif english_ratio > 0.6:
            # 英美式区分
            text_lower = text_cleaned.lower()
            british_count = sum(1 for word in LanguageDetector.BRITISH_PATTERNS 
                              if re.search(r'\b' + word + r'\b', text_lower))
            american_count = sum(1 for word in LanguageDetector.AMERICAN_PATTERNS 
                               if re.search(r'\b' + word + r'\b', text_lower))
            
            # 只有明确检测到拼写差异时才区分英美式
            if british_count > american_count and british_count >= 2:
                return 'eng_gb'
            elif american_count > british_count and american_count >= 2:
                return 'eng_us'
            else:
                return 'eng_us'  # 默认美式
        
        # 策略3: 比例相近（40%-60%之间），看谁更多
        elif chinese_ratio > english_ratio:
            return 'zho_cn'
        else:
            return 'eng_us'
    
    @staticmethod
    def get_language_name(lang_code: str) -> str:
        """获取语言名称"""
        names = {
            'zho_cn': '简体中文',
            'zho_tw': '繁体中文',
            'eng_us': 'English (US)',
            'eng_gb': 'English (UK)'
        }
        return names.get(lang_code, '未知语言')


class DocumentAnalyzer:
    """文档分析核心引擎"""
    
    def __init__(self):
        self.main = MockMain()
        logger.info("文档分析器初始化完成")
    
    def analyze_text(self, text: str, language: Optional[str] = None) -> dict:
        """
        分析文本
        
        Args:
            text: 要分析的文本
            language: 指定语言，如果为None则自动检测
            
        Returns:
            分析结果字典
        """
        if not text or not text.strip():
            raise ValueError("文本不能为空")
        
        # 语言检测
        if language is None:
            language = LanguageDetector.detect_language(text)
            if not language:
                raise ValueError("无法识别文档语言")
        
        logger.info(f"分析文本，语言: {language}, 长度: {len(text)}")
        
        # 句子分词
        sentences = wl_sentence_tokenization.wl_sentence_tokenize(
            self.main, text, lang=language
        )
        
        # 词语分词（同时记录每个句子的词数）
        tokens = []
        sentence_lengths = []
        for sentence in sentences:
            sentence_tokens = wl_word_tokenization.wl_word_tokenize_flat(
                self.main, sentence, lang=language
            )
            tokens.extend(sentence_tokens)
            sentence_lengths.append(len(sentence_tokens))
        
        tokens_text = [str(token) for token in tokens]
        
        # 计算指标
        return self._calculate_metrics(tokens_text, sentence_lengths, language)
    
    def _calculate_metrics(self, tokens_text: list, sentence_lengths: list, language: str) -> dict:
        """计算所有指标"""
        num_sentences = len(sentence_lengths)
        num_words = len(tokens_text)
        num_chars = sum(len(token) for token in tokens_text)
        
        if num_sentences == 0 or num_words == 0:
            raise ValueError("文本过短，无法分析")
        
        # 1. 可读性指标
        ari = 4.71 * (num_chars / num_words) + 0.5 * (num_words / num_sentences) - 21.43
        long_words = sum(1 for token in tokens_text if len(token) > 6)
        lix = (num_words / num_sentences) + (long_words * 100 / num_words)
        
        L = (num_chars / num_words) * 100
        S = (num_sentences / num_words) * 100
        cli = 0.0588 * L - 0.296 * S - 15.8
        
        # 2. 词汇多样性指标
        num_types = len(set(tokens_text))
        ttr = num_types / num_words
        rttr = num_types / np.sqrt(num_words)
        cttr = num_types / np.sqrt(2 * num_words)
        herdan_c = np.log(num_types) / np.log(num_words) if num_words > 1 else 0
        
        # Yule's K
        tokens_freq = Counter(tokens_text)
        freqs_count = Counter(tokens_freq.values())
        s2 = sum(freq ** 2 * count for freq, count in freqs_count.items())
        yule_k = 10000 * (s2 - num_words) / (num_words ** 2) if num_words > 0 else 0
        
        # 3. 结构复杂度指标
        word_lengths = [len(token) for token in tokens_text]
        
        # 4. 词频统计（过滤标点）
        tokens_freq_filtered = Counter([
            token for token in tokens_text 
            if any(c.isalnum() for c in token)
        ])
        top_words = tokens_freq_filtered.most_common(10)
        
        # 构建结果
        results = {
            'language': {
                'detected': language,
                'name': LanguageDetector.get_language_name(language)
            },
            'readability': {
                'ARI': round(ari, 2),
                'Lix': round(lix, 2),
                'Coleman_Liau_Index': round(cli, 2),
                'interpretation': self._interpret_readability(ari, lix)
            },
            'lexical_diversity': {
                'num_tokens': num_words,
                'num_types': num_types,
                'TTR': round(ttr, 4),
                'RTTR': round(rttr, 4),
                'CTTR': round(cttr, 4),
                'Herdan_C': round(herdan_c, 4),
                'Yule_K': round(yule_k, 2),
                'interpretation': self._interpret_diversity(ttr)
            },
            'structural_complexity': {
                'num_sentences': num_sentences,
                'num_words': num_words,
                'num_chars': num_chars,
                'avg_sentence_length': round(np.mean(sentence_lengths), 2),
                'std_sentence_length': round(np.std(sentence_lengths), 2),
                'max_sentence_length': int(np.max(sentence_lengths)),
                'min_sentence_length': int(np.min(sentence_lengths)),
                'avg_word_length': round(np.mean(word_lengths), 2),
                'std_word_length': round(np.std(word_lengths), 2),
                'interpretation': self._interpret_complexity(
                    np.mean(sentence_lengths), language
                )
            },
            'top_words': [(word, count) for word, count in top_words]
        }
        
        return results
    
    def _interpret_readability(self, ari: float, lix: float) -> str:
        """解释可读性"""
        if ari < 10 and lix < 40:
            return "易读 - 适合大众读者"
        elif ari < 14 and lix < 50:
            return "中等 - 适合高中及以上读者"
        else:
            return "困难 - 适合专业读者"
    
    def _interpret_diversity(self, ttr: float) -> str:
        """解释词汇多样性"""
        if ttr > 0.6:
            return "丰富 - 词汇使用多样"
        elif ttr > 0.5:
            return "中等 - 词汇使用适中"
        else:
            return "重复 - 词汇重复较多"
    
    def _interpret_complexity(self, avg_sent_len: float, language: str) -> str:
        """解释结构复杂度"""
        if 'zho' in language:
            if avg_sent_len < 20:
                return "简单 - 句子结构清晰"
            elif avg_sent_len < 30:
                return "中等 - 句子结构适中"
            else:
                return "复杂 - 句子较长"
        else:
            if avg_sent_len < 15:
                return "简单 - 句子结构清晰"
            elif avg_sent_len < 20:
                return "中等 - 句子结构适中"
            else:
                return "复杂 - 句子较长"
    
    def format_results_as_text(self, results: dict) -> str:
        """将结果格式化为人类可读的文本"""
        lines = [
            "=" * 60,
            "  智能文档分析报告",
            "=" * 60,
            ""
        ]
        
        # 语言信息
        lang_info = results['language']
        struct_info = results['structural_complexity']
        lines.extend([
            f"📌 检测语言: {lang_info['name']}",
            f"   文档规模: {struct_info['num_words']} 词, {struct_info['num_sentences']} 句",
            ""
        ])
        
        # 可读性
        read_info = results['readability']
        lines.extend([
            "📖 可读性分析:",
            f"   ARI指数: {read_info['ARI']}",
            f"   Lix指数: {read_info['Lix']}",
            f"   Coleman-Liau: {read_info['Coleman_Liau_Index']}",
            f"   💡 {read_info['interpretation']}",
            ""
        ])
        
        # 词汇多样性
        lex_info = results['lexical_diversity']
        lines.extend([
            "📚 词汇多样性:",
            f"   词型数/词符数: {lex_info['num_types']}/{lex_info['num_tokens']}",
            f"   TTR: {lex_info['TTR']}",
            f"   RTTR: {lex_info['RTTR']}",
            f"   Herdan's C: {lex_info['Herdan_C']}",
            f"   💡 {lex_info['interpretation']}",
            ""
        ])
        
        # 结构复杂度
        lines.extend([
            "🔍 结构复杂度:",
            f"   平均句长: {struct_info['avg_sentence_length']} 词",
            f"   句长标准差: {struct_info['std_sentence_length']}",
            f"   平均词长: {struct_info['avg_word_length']} 字符",
            f"   💡 {struct_info['interpretation']}",
            ""
        ])
        
        # 高频词
        lines.append("📈 高频词 Top 10:")
        for i, (word, count) in enumerate(results['top_words'], 1):
            lines.append(f"   {i:2d}. {word:<15} ({count} 次)")
        
        lines.extend(["", "=" * 60])
        
        return "\n".join(lines)


# MCP服务器实例（在main中初始化）
mcp: Optional[FastMCP] = None

# 全局分析器实例（延迟初始化）
_analyzer: Optional[DocumentAnalyzer] = None


def get_analyzer() -> DocumentAnalyzer:
    """获取分析器实例（延迟初始化）"""
    global _analyzer
    if _analyzer is None:
        _analyzer = DocumentAnalyzer()
    return _analyzer


def to_json(data: dict) -> str:
    """统一的JSON序列化函数"""
    return json.dumps(data, ensure_ascii=False, indent=2)


def validate_text(text: str) -> None:
    """验证文本输入"""
    if not text or not text.strip():
        raise ValueError("text参数不能为空")


def create_mcp_server() -> FastMCP:
    """
    创建并配置MCP服务器
    
    Returns:
        配置好的FastMCP服务器实例
    """
    # 配置传输安全
    transport_security = TransportSecuritySettings(
        enable_dns_rebinding_protection=False,
        allowed_hosts=["*"],
        allowed_origins=["*"]
    )
    
    # 创建服务器
    server = FastMCP(
        "wordless-doc-analyzer",
        transport_security=transport_security
    )
    
    # 注册工具
    @server.tool()
    def analyze_document(
        text: str,
        language: Optional[str] = None,
        format: str = "json"
    ) -> str:
        """
        分析文档文本，提供可读性、词汇多样性和结构复杂度分析。
        自动检测中英文，支持简体中文、繁体中文、英式英语和美式英语。
        
        Args:
            text: 要分析的文档文本内容
            language: 指定文档语言（可选），支持: zho_cn, zho_tw, eng_us, eng_gb
            format: 输出格式，json（结构化数据）或 text（人类可读），默认json
        """
        validate_text(text)
        
        analyzer = get_analyzer()
        results = analyzer.analyze_text(text, language)
        
        return analyzer.format_results_as_text(results) if format == "text" else to_json(results)
    
    @server.tool()
    def detect_language(text: str) -> str:
        """
        检测文本的语言类型（中文/英文，简体/繁体，美式/英式）
        
        Args:
            text: 要检测的文本
        """
        validate_text(text)
        
        lang = LanguageDetector.detect_language(text)
        if not lang:
            return to_json({"error": "无法识别语言"})
        
        return to_json({
            "language_code": lang,
            "language_name": LanguageDetector.get_language_name(lang)
        })
    
    # 注册资源
    @server.resource("doc://supported-languages")
    def get_supported_languages() -> str:
        """获取支持的语言列表"""
        return to_json({
            "supported_languages": [
                {"code": "zho_cn", "name": "简体中文", "description": "Simplified Chinese"},
                {"code": "zho_tw", "name": "繁体中文", "description": "Traditional Chinese"},
                {"code": "eng_us", "name": "English (US)", "description": "American English"},
                {"code": "eng_gb", "name": "English (UK)", "description": "British English"}
            ],
            "auto_detection": True,
            "metrics": {
                "readability": ["ARI", "Lix", "Coleman-Liau Index"],
                "lexical_diversity": ["TTR", "RTTR", "CTTR", "Herdan's C", "Yule's K"],
                "structural_complexity": ["句长统计", "词长统计"]
            }
        })
    
    @server.resource("doc://analysis-metrics")
    def get_analysis_metrics() -> str:
        """获取所有支持的分析指标说明"""
        return to_json({
            "readability_metrics": {
                "ARI": {
                    "name": "Automated Readability Index",
                    "description": "基于字符数和词数的可读性指数",
                    "interpretation": "< 10: 易读, 10-14: 中等, > 14: 困难"
                },
                "Lix": {
                    "name": "Läsbarhetsindex",
                    "description": "瑞典可读性指数，考虑长词比例",
                    "interpretation": "< 40: 易读, 40-50: 中等, > 50: 困难"
                },
                "Coleman_Liau_Index": {
                    "name": "Coleman-Liau Index",
                    "description": "基于字符和句子的可读性指数"
                }
            },
            "diversity_metrics": {
                "TTR": {
                    "name": "Type-Token Ratio",
                    "description": "词型与词符的比率，衡量词汇丰富度",
                    "interpretation": "> 0.6: 丰富, 0.5-0.6: 中等, < 0.5: 重复"
                },
                "RTTR": {
                    "name": "Root Type-Token Ratio",
                    "description": "词型数除以词符数的平方根"
                },
                "CTTR": {
                    "name": "Corrected Type-Token Ratio",
                    "description": "修正的TTR，更稳定"
                },
                "Herdan_C": {
                    "name": "Herdan's C",
                    "description": "对数形式的词汇丰富度指标"
                },
                "Yule_K": {
                    "name": "Yule's K",
                    "description": "基于词频分布的词汇多样性指标"
                }
            },
            "structural_metrics": {
                "avg_sentence_length": "平均句子长度（词数）",
                "avg_word_length": "平均词长（字符数）",
                "sentence_length_std": "句长标准差"
            }
        })
    
    # 注册提示模板
    @server.prompt()
    def analyze_document_prompt(text_sample: str) -> str:
        """
        生成文档分析提示词
        
        Args:
            text_sample: 文档样本文本（可以是完整文档或摘要）
        """
        return f"""请使用 analyze_document 工具分析以下文本：

文本内容：
{text_sample}

分析要求：
1. 自动检测语言（支持中英文）
2. 评估可读性水平（ARI、Lix等指标）
3. 分析词汇多样性（TTR、RTTR等）
4. 评估结构复杂度（句长、词长等）
5. 提供人类可读的解释

请调用工具并解读结果。"""
    
    @server.prompt()
    def compare_documents_prompt(text1: str, text2: str, aspect: str = "overall") -> str:
        """
        生成文档对比分析提示词
        
        Args:
            text1: 第一篇文档
            text2: 第二篇文档
            aspect: 对比维度（overall/readability/diversity/structure）
        """
        aspects_desc = {
            "overall": "所有指标",
            "readability": "可读性",
            "diversity": "词汇多样性",
            "structure": "结构复杂度"
        }
        
        return f"""请对比分析以下两篇文档的{aspects_desc.get(aspect, '所有指标')}：

【文档1】
{text1[:500]}...

【文档2】
{text2[:500]}...

分析步骤：
1. 分别使用 analyze_document 工具分析两篇文档
2. 对比关键指标差异
3. 解释差异的实际意义
4. 给出改进建议（如果适用）

请开始分析。"""
    
    @server.prompt()
    def readability_improvement_prompt(text: str) -> str:
        """
        生成可读性改进建议提示词
        
        Args:
            text: 需要改进的文档文本
        """
        return f"""请分析以下文档的可读性，并提供改进建议：

文档内容：
{text}

分析流程：
1. 使用 analyze_document 工具获取详细指标
2. 识别可读性问题（句子过长、词汇过于复杂等）
3. 提供具体改进建议：
   - 句子长度优化
   - 词汇选择建议
   - 结构调整方案
4. 如果可能，给出改写示例

请开始分析并提供建议。"""
    
    return server




def configure_http_server(host: str = "0.0.0.0", port: int = 8000):
    """
    配置FastMCP的HTTP服务器参数
    
    Args:
        host: 服务器地址（默认: 0.0.0.0，监听所有网络接口）
        port: 服务器端口（默认: 8000）
    """
    mcp.settings.host = host
    mcp.settings.port = port
    logger.info(f"🚀 HTTP服务器配置:")
    logger.info(f"   监听地址: {host}")
    logger.info(f"   监听端口: {port}")
    logger.info(f"   访问端点: http://{host}:{port}{mcp.settings.streamable_http_path}")
    
    # 安全提示
    if host == "0.0.0.0":
        logger.warning(f"⚠️  服务器监听所有网络接口，可从任何IP访问")
        logger.warning(f"⚠️  生产环境建议配置防火墙和访问控制")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='智能文档分析器 MCP 服务器 - 提供文档可读性、词汇多样性和结构复杂度分析',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
功能说明:
  Tools (工具):
    • analyze_document  - 完整的文档分析（可读性、词汇多样性、结构复杂度）
    • detect_language   - 智能语言检测（支持中英文）
  
  Resources (资源):
    • doc://supported-languages - 支持的语言列表
    • doc://analysis-metrics    - 分析指标详细说明
  
  Prompts (提示模板):
    • analyze_document_prompt           - 文档分析提示
    • compare_documents_prompt          - 文档对比分析提示
    • readability_improvement_prompt    - 可读性改进建议提示

认证说明:
  使用 --auth-token 或环境变量 MCP_AUTH_TOKEN 启用Bearer Token认证
  客户端需要在请求头中添加: Authorization: Bearer <token>

示例:
  stdio模式:     python mcp_doc_analyzer_server.py
  HTTP模式:      python mcp_doc_analyzer_server.py --transport http --host 0.0.0.0 --port 8000
  启用认证:      python mcp_doc_analyzer_server.py --transport http --auth-token your-secret-token
  环境变量认证:  MCP_AUTH_TOKEN=your-token python mcp_doc_analyzer_server.py --transport http
        """
    )
    parser.add_argument(
        '--transport',
        choices=['stdio', 'http', 'streamable-http'],
        default='stdio',
        help='传输模式（默认: stdio）'
    )
    parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='HTTP服务器地址（默认: 0.0.0.0，监听所有接口）'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='HTTP服务器端口（默认: 8000）'
    )
    parser.add_argument(
        '--auth-token',
        type=str,
        default=None,
        help='认证token（Bearer Token）。也可通过环境变量MCP_AUTH_TOKEN设置'
    )
    
    args = parser.parse_args()
    
    try:
        # 创建MCP服务器
        global mcp
        mcp = create_mcp_server()
        
        # 获取认证token
        auth_token = args.auth_token or os.getenv('MCP_AUTH_TOKEN')
        
        if args.transport in ['http', 'streamable-http']:
            logger.info("🌐 准备启动远程HTTP服务...")
            configure_http_server(args.host, args.port)
            
            # 如果设置了认证token，包装应用
            if auth_token:
                # 获取FastMCP的ASGI应用
                app = mcp.streamable_http_app()
                # 使用认证中间件包装
                wrapped_app = BearerTokenMiddleware(app, auth_token)
                # 手动启动uvicorn
                import uvicorn
                config = uvicorn.Config(
                    wrapped_app,
                    host=args.host,
                    port=args.port,
                    log_level="info"
                )
                server_instance = uvicorn.Server(config)
                import anyio
                anyio.run(server_instance.serve)
            else:
                logger.warning("⚠️  未设置认证token，服务器无需认证")
                mcp.run(transport='streamable-http')
        else:
            logger.info("🚀 启动 stdio 模式")
            if auth_token:
                logger.warning("⚠️  stdio模式不支持认证，忽略--auth-token参数")
            mcp.run()
    except KeyboardInterrupt:
        logger.info("\n✅ 服务器已停止")
    except Exception as e:
        logger.error(f"❌ 服务器错误: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()

