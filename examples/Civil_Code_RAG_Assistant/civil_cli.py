"""
民法典RAG助手 - 命令行版本
"""

import sys
import os
import readline  # 用于命令行历史记录
from pathlib import Path

# 添加模块路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from examples.Civil_Code_RAG_Assistant.configs.civil_config import (
    print_config_summary,
    CIVIL_DATA_DIR
)
from examples.Civil_Code_RAG_Assistant.rag.civil_rag import CivilCodeRAG

class CivilCodeCLI:
    """命令行交互界面"""
    
    def __init__(self):
        self.rag = None
        self.history = []
        self.running = True
        
    def initialize(self):
        """初始化系统"""
        print("=" * 70)
        print("⚖️  民法典智能助手 - 命令行版本")
        print("=" * 70)
        
        # 显示配置
        print_config_summary()
        
        print("\n🔧 初始化系统...")
        try:
            self.rag = CivilCodeRAG(verbose=True)
            self.rag.initialize()
            print("✅ 系统初始化完成！")
            print("💡 输入 'help' 查看帮助，'quit' 退出")
        except Exception as e:
            print(f"❌ 初始化失败: {e}")
            print("请检查：")
            print("  1. 是否已运行入库程序？")
            print("  2. Ollama服务是否正常？")
            sys.exit(1)
    
    def show_help(self):
        """显示帮助信息"""
        help_text = """
📚 命令列表：
  help              - 显示此帮助信息
  quit / exit       - 退出程序
  clear / cls       - 清屏
  history           - 显示查询历史
  stats             - 显示系统统计
  example           - 显示示例问题
  test              - 运行测试查询
  about             - 关于本系统

📝 直接输入法律问题即可获得回答，例如：
  1. 租房合同违约怎么办？
  2. 离婚财产如何分割？
  3. 被狗咬了怎么赔偿？
  4. 合同无效的情形有哪些？
  5. 继承遗产需要什么手续？
        """
        print(help_text)
    
    def show_examples(self):
        """显示示例问题"""
        examples = [
            "什么是违约责任？",
            "离婚需要什么条件？",
            "合同无效的情况有哪些？",
            "个人隐私权受到侵害怎么办？",
            "交通事故责任如何认定？",
            "房屋租赁合同要注意什么？",
            "遗嘱怎么写才有效？",
            "消费者权益受到侵害如何维权？",
            "产品质量问题怎么赔偿？",
            "夫妻共同债务如何认定？"
        ]
        
        print("\n📋 示例法律问题：")
        for i, example in enumerate(examples, 1):
            print(f"  {i:2d}. {example}")
        print("\n💡 输入问题编号或直接输入您的问题")
    
    def show_stats(self):
        """显示系统统计"""
        if not self.rag:
            print("❌ 系统未初始化")
            return
        
        info = self.rag.get_system_info()
        print("\n📊 系统统计：")
        print(f"  状态: {info['status']}")
        print(f"  向量库条目: {info['vector_count']}")
        print(f"  嵌入模型: {info['embedding_model']}")
        print(f"  LLM模型: {info['llm_model']}")
        print(f"  检索方法: {info['retrieval_method']}")
        print(f"  温度参数: {info['temperature']}")
    
    def show_history(self):
        """显示查询历史"""
        if not self.history:
            print("📭 暂无查询历史")
            return
        
        print("\n📜 查询历史：")
        for i, item in enumerate(self.history[-10:], 1):  # 显示最近10条
            question, timestamp = item
            print(f"  {i:2d}. [{timestamp}] {question[:50]}...")
    
    def clear_screen(self):
        """清屏"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def run_test(self):
        """运行测试查询"""
        test_questions = [
            "什么是违约责任？",
            "离婚后财产怎么分割？"
        ]
        
        print("\n🧪 运行测试查询...")
        for question in test_questions:
            print(f"\n{'='*60}")
            print(f"测试: {question}")
            print(f"{'='*60}")
            
            # 先显示检索结果
            try:
                docs = self.rag.get_retrieved_documents(question, k=2)
                print(f"🔍 检索到 {len(docs)} 个相关法条：")
                for i, doc in enumerate(docs, 1):
                    article = doc.metadata.get('article_number', '未知法条')
                    preview = doc.page_content[:80].replace('\n', ' ')
                    print(f"  {i}. {article}: {preview}...")
            except Exception as e:
                print(f"⚠️  检索时出错: {e}")
            
            # 获取完整回答
            print(f"\n🤖 AI回答：")
            response = self.rag.query_simple(question)
            print(response)
    
    def show_about(self):
        """显示关于信息"""
        about_text = """
══════════════════════════════════════════════════════════════════════
                        民法典智能助手 - 命令行版本
══════════════════════════════════════════════════════════════════════

📖 系统简介：
  基于RAG技术的《中华人民共和国民法典》智能问答系统，能够准确检索
  并解释民法典相关条文，提供专业的法律咨询。

🏗️  技术架构：
  • 向量检索：ChromaDB + Ollama Embeddings
  • 大语言模型：Ollama本地模型
  • 文档处理：LangChain + Unstructured

🔐 特点：
  • 完全本地运行，保护隐私
  • 实时检索最新民法典条文
  • 专业的法律解释和建议
  • 完全免费，无使用限制

⚖️  免责声明：
  本系统提供的法律信息仅供参考，不构成正式法律意见。
  具体案件请咨询专业律师。

══════════════════════════════════════════════════════════════════════
        """
        print(about_text)
    
    def process_command(self, command: str):
        """处理用户命令"""
        command = command.strip().lower()
        
        if command in ['quit', 'exit', 'q']:
            print("\n👋 感谢使用民法典智能助手，再见！")
            self.running = False
        
        elif command in ['help', '?']:
            self.show_help()
        
        elif command in ['clear', 'cls']:
            self.clear_screen()
        
        elif command == 'history':
            self.show_history()
        
        elif command == 'stats':
            self.show_stats()
        
        elif command == 'example':
            self.show_examples()
        
        elif command == 'test':
            self.run_test()
        
        elif command == 'about':
            self.show_about()
        
        elif command.isdigit():
            # 输入数字，选择示例问题
            try:
                idx = int(command) - 1
                examples = [
                    "什么是违约责任？",
                    "离婚需要什么条件？",
                    "合同无效的情况有哪些？",
                    "个人隐私权受到侵害怎么办？",
                    "交通事故责任如何认定？"
                ]
                if 0 <= idx < len(examples):
                    self.process_query(examples[idx])
                else:
                    print(f"❌ 请输入1-{len(examples)}之间的数字")
            except ValueError:
                print("❌ 无效的数字")
        
        elif command:
            # 普通查询
            self.process_query(command)
    
    def process_query(self, question: str):
        """处理法律查询"""
        import time
        from datetime import datetime
        
        if not question or len(question) < 2:
            print("❌ 问题太短，请详细描述")
            return
        
        print(f"\n🔍 正在查询: {question}")
        start_time = time.time()
        
        try:
            # 获取回答
            response = self.rag.query_simple(question)
            
            # 显示结果
            print("\n" + "=" * 70)
            print("⚖️  法律咨询结果")
            print("=" * 70)
            print(response)
            print("=" * 70)
            
            # 计算耗时
            elapsed = time.time() - start_time
            print(f"⏱️  查询耗时: {elapsed:.2f}秒")
            
            # 保存到历史
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.history.append((question, timestamp))
            
        except Exception as e:
            print(f"❌ 查询失败: {e}")
            import traceback
            traceback.print_exc()
    
    def run(self):
        """运行主循环"""
        self.initialize()
        
        # 设置命令行历史
        try:
            readline.read_history_file(".civil_history")
        except FileNotFoundError:
            pass
        
        # 主循环
        while self.running:
            try:
                # 显示提示符
                print("\n" + "─" * 50)
                user_input = input("💬 请输入问题或命令: ").strip()
                
                if user_input:
                    self.process_command(user_input)
            
            except KeyboardInterrupt:
                print("\n\n⚠️  检测到Ctrl+C，输入 'quit' 退出程序")
            
            except EOFError:
                print("\n👋 感谢使用，再见！")
                self.running = False
        
        # 保存历史
        try:
            readline.write_history_file(".civil_history")
        except:
            pass

def main():
    """主函数"""
    cli = CivilCodeCLI()
    cli.run()

if __name__ == "__main__":
    main()