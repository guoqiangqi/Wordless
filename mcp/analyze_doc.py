#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能文档分析器 - 基于Wordless核心模块
自动识别中英文，提供可读性、可用性、发现性分析

使用方法:
    python analyze_doc.py document.txt
    python analyze_doc.py document.txt --output results.json
"""

import argparse
import sys
import os
import json
import re
from pathlib import Path
from collections import Counter

# 添加Wordless模块到路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

try:
    # 检查spaCy
    import spacy
except ImportError:
    print(f"❌ 错误: 未安装spaCy")
    print(f"   请运行: pip install spacy")
    sys.exit(1)

try:
    from wordless.wl_nlp import (
        wl_sentence_tokenization,
        wl_word_tokenization,
        wl_texts
    )
    from wordless.wl_settings import wl_settings_global
    from wordless.wl_utils import wl_misc
except ImportError as e:
    print(f"❌ 错误: 无法导入Wordless模块")
    print(f"   请确保:")
    print(f"   1. 在Wordless根目录下运行此脚本")
    print(f"   2. 已安装Wordless的所有依赖")
    print(f"   3. 如果是开发版，请确保wordless目录结构完整")
    print(f"\n   详细错误: {e}")
    print(f"\n   提示: 这是Wordless的扩展工具，需要Wordless环境")
    sys.exit(1)


class MockMain:
    """模拟Wordless主窗口对象"""
    def __init__(self):
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
    """智能语言检测器"""
    
    @staticmethod
    def detect_language(text):
        """
        自动检测文本语言
        返回: 'zho_cn', 'zho_tw', 'eng_us', 'eng_gb' 或 None
        """
        # 统计字符类型
        total_chars = len(text)
        if total_chars == 0:
            return None
        
        # 统计中文字符（包括标点）
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        # 统计英文字母
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        
        chinese_ratio = chinese_chars / total_chars
        english_ratio = english_chars / total_chars
        
        # 判断逻辑
        if chinese_ratio > 0.3:  # 中文字符超过30%
            # 检测是简体还是繁体（简单判断）
            simplified_common = len(re.findall(r'[的是在了有和人这中国]', text))
            traditional_common = len(re.findall(r'[個們說處時間]', text))
            
            if traditional_common > simplified_common * 1.5:
                return 'zho_tw'
            else:
                return 'zho_cn'
                
        elif english_ratio > 0.5:  # 英文字母超过50%
            # 简单判断英式/美式（基于常见拼写差异）
            british_patterns = len(re.findall(r'\b(colour|favour|honour|centre|theatre)\b', text.lower()))
            american_patterns = len(re.findall(r'\b(color|favor|honor|center|theater)\b', text.lower()))
            
            if british_patterns > american_patterns:
                return 'eng_gb'
            else:
                return 'eng_us'
        
        # 默认返回英文
        return 'eng_us'
    
    @staticmethod
    def get_language_name(lang_code):
        """获取语言名称"""
        names = {
            'zho_cn': '简体中文',
            'zho_tw': '繁体中文',
            'eng_us': 'English (US)',
            'eng_gb': 'English (UK)'
        }
        return names.get(lang_code, '未知语言')


class DocumentAnalyzer:
    """智能文档分析器"""
    
    def __init__(self):
        """初始化分析器"""
        self.main = MockMain()
        self.results = {}
        self.lang = None
        self.text = None
        
    def load_file(self, file_path):
        """加载文件并自动检测语言"""
        try:
            # 尝试UTF-8编码
            with open(file_path, 'r', encoding='utf-8') as f:
                self.text = f.read().strip()
        except UnicodeDecodeError:
            # 尝试其他常见编码
            for encoding in ['gbk', 'gb2312', 'big5', 'latin1']:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        self.text = f.read().strip()
                    print(f"⚠️  检测到文件编码为: {encoding}")
                    break
                except:
                    continue
            else:
                print(f"❌ 无法读取文件，请确保文件编码正确")
                return False
        except Exception as e:
            print(f"❌ 加载文件失败: {e}")
            return False
        
        if not self.text:
            print(f"❌ 文件为空")
            return False
        
        # 自动检测语言
        self.lang = LanguageDetector.detect_language(self.text)
        if not self.lang:
            print(f"❌ 无法识别文档语言")
            return False
        
        return True
    
    def tokenize(self):
        """分词处理"""
        try:
            # 句子分词
            self.sentences = wl_sentence_tokenization.wl_sentence_tokenize(
                self.main, self.text, lang=self.lang
            )
            
            # 词语分词
            self.tokens = []
            for sentence in self.sentences:
                tokens = wl_word_tokenization.wl_word_tokenize_flat(
                    self.main, sentence, lang=self.lang
                )
                self.tokens.extend(tokens)
            
            self.tokens_text = [str(token) for token in self.tokens]
            return True
        except Exception as e:
            print(f"❌ 分词失败: {e}")
            print(f"   可能需要下载语言模型")
            if 'zho' in self.lang:
                print(f"   运行: python3 -m spacy download zh_core_web_lg")
            else:
                print(f"   运行: python3 -m spacy download en_core_web_lg")
            return False
    
    def calculate_metrics(self):
        """计算所有指标"""
        num_sentences = len(self.sentences)
        num_words = len(self.tokens)
        num_chars = sum(len(token) for token in self.tokens_text)
        
        if num_sentences == 0 or num_words == 0:
            print("⚠️  文本过短，无法分析")
            return False
        
        # 1. 可读性指标
        # ARI
        ari = 4.71 * (num_chars / num_words) + 0.5 * (num_words / num_sentences) - 21.43
        
        # Lix
        long_words = sum(1 for token in self.tokens_text if len(token) > 6)
        lix = (num_words / num_sentences) + (long_words * 100 / num_words)
        
        # Coleman-Liau
        L = (num_chars / num_words) * 100
        S = (num_sentences / num_words) * 100
        cli = 0.0588 * L - 0.296 * S - 15.8
        
        # 2. 词汇多样性指标
        num_types = len(set(self.tokens_text))
        ttr = num_types / num_words
        rttr = num_types / np.sqrt(num_words)
        cttr = num_types / np.sqrt(2 * num_words)
        
        if num_words > 1:
            herdan_c = np.log(num_types) / np.log(num_words)
        else:
            herdan_c = 0
        
        # Yule's K
        tokens_freq = Counter(self.tokens_text)
        freqs_count = Counter(tokens_freq.values())
        s2 = sum(freq ** 2 * count for freq, count in freqs_count.items())
        yule_k = 10000 * (s2 - num_words) / (num_words ** 2) if num_words > 0 else 0
        
        # 3. 结构复杂度指标
        sentence_lengths = []
        for sentence in self.sentences:
            tokens = wl_word_tokenization.wl_word_tokenize_flat(
                self.main, sentence, lang=self.lang
            )
            sentence_lengths.append(len(tokens))
        
        word_lengths = [len(token) for token in self.tokens_text]
        
        # 4. 词频统计 - 过滤标点符号
        # 只统计包含字母或数字的词
        tokens_freq_filtered = Counter([
            token for token in self.tokens_text 
            if any(c.isalnum() for c in token)  # 至少包含一个字母或数字
        ])
        top_words = tokens_freq_filtered.most_common(20)
        
        # 保存结果
        self.results = {
            'language': {
                'detected': self.lang,
                'name': LanguageDetector.get_language_name(self.lang)
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
                'interpretation': self._interpret_complexity(np.mean(sentence_lengths))
            },
            'top_words': [(word, count) for word, count in top_words[:10]]
        }
        
        return True
    
    def _interpret_readability(self, ari, lix):
        """解释可读性"""
        if ari < 10 and lix < 40:
            return "易读 - 适合大众读者"
        elif ari < 14 and lix < 50:
            return "中等 - 适合高中及以上读者"
        else:
            return "困难 - 适合专业读者"
    
    def _interpret_diversity(self, ttr):
        """解释词汇多样性"""
        if ttr > 0.6:
            return "丰富 - 词汇使用多样"
        elif ttr > 0.5:
            return "中等 - 词汇使用适中"
        else:
            return "重复 - 词汇重复较多"
    
    def _interpret_complexity(self, avg_sent_len):
        """解释结构复杂度"""
        # 根据语言调整标准
        if 'zho' in self.lang:
            if avg_sent_len < 20:
                return "简单 - 句子结构清晰"
            elif avg_sent_len < 30:
                return "中等 - 句子结构适中"
            else:
                return "复杂 - 句子较长"
        else:  # 英文
            if avg_sent_len < 15:
                return "简单 - 句子结构清晰"
            elif avg_sent_len < 20:
                return "中等 - 句子结构适中"
            else:
                return "复杂 - 句子较长"
    
    def print_results(self):
        """打印分析结果"""
        print(f"\n{'='*60}")
        print(f"  智能文档分析报告")
        print(f"{'='*60}\n")
        
        # 语言信息
        print(f"📌 检测语言: {self.results['language']['name']}")
        print(f"   文档规模: {self.results['structural_complexity']['num_words']} 词, "
              f"{self.results['structural_complexity']['num_sentences']} 句\n")
        
        # 可读性
        print(f"📖 可读性分析:")
        print(f"   ARI指数: {self.results['readability']['ARI']}")
        print(f"   Lix指数: {self.results['readability']['Lix']}")
        print(f"   Coleman-Liau: {self.results['readability']['Coleman_Liau_Index']}")
        print(f"   💡 {self.results['readability']['interpretation']}\n")
        
        # 词汇多样性
        print(f"📚 词汇多样性:")
        print(f"   词型数/词符数: {self.results['lexical_diversity']['num_types']}/"
              f"{self.results['lexical_diversity']['num_tokens']}")
        print(f"   TTR: {self.results['lexical_diversity']['TTR']}")
        print(f"   RTTR: {self.results['lexical_diversity']['RTTR']}")
        print(f"   Herdan's C: {self.results['lexical_diversity']['Herdan_C']}")
        print(f"   💡 {self.results['lexical_diversity']['interpretation']}\n")
        
        # 结构复杂度
        print(f"🔍 结构复杂度:")
        print(f"   平均句长: {self.results['structural_complexity']['avg_sentence_length']} 词")
        print(f"   句长标准差: {self.results['structural_complexity']['std_sentence_length']}")
        print(f"   平均词长: {self.results['structural_complexity']['avg_word_length']} 字符")
        print(f"   💡 {self.results['structural_complexity']['interpretation']}\n")
        
        # 高频词
        print(f"📈 高频词 Top 10:")
        for i, (word, count) in enumerate(self.results['top_words'], 1):
            print(f"   {i:2d}. {word:<15} ({count} 次)")
        
        print(f"\n{'='*60}\n")
    
    def save_results(self, output_path):
        """保存结果为JSON"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, ensure_ascii=False, indent=2)
            print(f"✅ 结果已保存至: {output_path}")
        except Exception as e:
            print(f"❌ 保存失败: {e}")
    
    def analyze(self, file_path, output_json=None):
        """执行完整分析"""
        print(f"\n🚀 开始分析文档: {file_path}\n")
        
        # 加载文件
        if not self.load_file(file_path):
            return False
        
        print(f"✅ 文件加载成功 ({len(self.text)} 字符)")
        print(f"✅ 语言识别: {LanguageDetector.get_language_name(self.lang)}\n")
        
        # 分词
        print(f"📝 正在分词...")
        if not self.tokenize():
            return False
        
        print(f"✅ 分词完成: {len(self.sentences)} 个句子, {len(self.tokens)} 个词\n")
        
        # 计算指标
        print(f"📊 正在计算指标...")
        if not self.calculate_metrics():
            return False
        
        print(f"✅ 计算完成")
        
        # 显示结果
        self.print_results()
        
        # 保存JSON
        if output_json:
            self.save_results(output_json)
        
        return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='智能文档分析器 - 自动识别中英文',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  基础使用（自动识别语言）:
    python analyze_doc.py document.txt
    
  保存JSON结果:
    python analyze_doc.py document.txt --output results.json
    python analyze_doc.py document.txt -o results.json

支持的语言:
  - 简体中文 (自动识别)
  - 繁体中文 (自动识别)
  - English US (自动识别)
  - English UK (自动识别)

分析指标:
  📖 可读性: ARI, Lix, Coleman-Liau指数
  📚 词汇多样性: TTR, RTTR, CTTR, Herdan's C, Yule's K
  🔍 结构复杂度: 句长、词长统计
  📈 词频分析: Top 10 高频词
        """
    )
    
    parser.add_argument(
        'file',
        help='要分析的文档文件路径'
    )
    
    parser.add_argument(
        '--output', '-o',
        help='输出JSON结果文件路径（可选）'
    )
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.file):
        print(f"❌ 错误: 文件不存在: {args.file}")
        return 1
    
    # 创建分析器并执行分析
    analyzer = DocumentAnalyzer()
    success = analyzer.analyze(args.file, output_json=args.output)
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())

