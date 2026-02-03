#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCP远程客户端示例
连接到远程HTTP MCP服务器并调用文档分析工具

使用方法:
    python mcp_remote_client.py --host localhost --port 8000
"""

import argparse
import asyncio
import json
import os

import httpx

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


async def test_remote_server(host: str = "localhost", port: int = 8000, auth_token: str = None):
    """
    测试远程MCP服务器
    
    Args:
        host: 服务器地址
        port: 服务器端口
        auth_token: 认证token（可选）
    """
    url = f"http://{host}:{port}/mcp"
    print(f"🔗 连接到远程服务器: {url}")
    
    if auth_token:
        print(f"🔐 使用Bearer Token认证")
    
    try:
        # 创建HTTP客户端，配置认证
        headers = {}
        if auth_token:
            headers['Authorization'] = f'Bearer {auth_token}'
        
        http_client = httpx.AsyncClient(headers=headers)
        
        async with streamable_http_client(url, http_client=http_client) as (read, write, _):
            async with ClientSession(read, write) as session:
                # 初始化会话
                await session.initialize()
                print("✅ 连接成功！")
                
                # 列出可用工具
                print("\n📋 可用工具:")
                tools = await session.list_tools()
                for tool in tools.tools:
                    print(f"  • {tool.name}: {tool.description}")
                
                # 列出可用资源
                print("\n📚 可用资源:")
                resources = await session.list_resources()
                for resource in resources.resources:
                    print(f"  • {resource.uri}: {resource.name}")
                
                # 列出可用提示模板
                print("\n💡 可用提示模板:")
                prompts = await session.list_prompts()
                for prompt in prompts.prompts:
                    print(f"  • {prompt.name}: {prompt.description}")
                
                # 测试语言检测
                print("\n🧪 测试1: 语言检测")
                test_text = "人工智能正在改变世界。"
                result = await session.call_tool("detect_language", {"text": test_text})
                print(f"  文本: {test_text}")
                for content in result.content:
                    if hasattr(content, 'text'):
                        data = json.loads(content.text)
                        print(f"  结果: {data['language_name']} ({data['language_code']})")
                
                # 测试文档分析
                print("\n🧪 测试2: 文档分析")
                test_doc = """
                人工智能（Artificial Intelligence，简称AI）是计算机科学的一个分支，
                它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。
                该领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。
                """
                result = await session.call_tool(
                    "analyze_document", 
                    {"text": test_doc, "format": "text"}
                )
                print("  分析结果:")
                for content in result.content:
                    if hasattr(content, 'text'):
                        print(content.text)
                
                # 读取资源
                print("\n🧪 测试3: 读取资源")
                resource_result = await session.read_resource("doc://supported-languages")
                for content in resource_result.contents:
                    if hasattr(content, 'text'):
                        data = json.loads(content.text)
                        print(f"  支持的语言:")
                        for lang in data['supported_languages']:
                            print(f"    • {lang['name']} ({lang['code']})")
                
                print("\n✨ 所有测试完成！")
                
    except Exception as e:
        print(f"❌ 错误: {e}")
        raise


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='MCP远程客户端 - 连接并测试远程MCP服务器',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  连接本地服务器:  python mcp_remote_client.py
  连接远程服务器:  python mcp_remote_client.py --host 192.168.1.100 --port 8000
  使用认证:        python mcp_remote_client.py --auth-token your-secret-token
  环境变量认证:    MCP_AUTH_TOKEN=your-token python mcp_remote_client.py
        """
    )
    parser.add_argument(
        '--host',
        type=str,
        default='localhost',
        help='服务器地址（默认: localhost）'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='服务器端口（默认: 8000）'
    )
    parser.add_argument(
        '--auth-token',
        type=str,
        default=None,
        help='认证token（Bearer Token）。也可通过环境变量MCP_AUTH_TOKEN设置'
    )
    
    args = parser.parse_args()
    
    try:
        auth_token = args.auth_token or os.getenv('MCP_AUTH_TOKEN')
        asyncio.run(test_remote_server(args.host, args.port, auth_token))
    except KeyboardInterrupt:
        print("\n✅ 客户端已停止")
    except Exception as e:
        print(f"❌ 客户端错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

