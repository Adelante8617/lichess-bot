"""所有测试模块最先导入它：把项目根加入 sys.path、切到临时目录、钉死开关。

必须在任何 bot 模块导入之前执行：bot.config 导入时就读取环境变量，load_dotenv 还会向上找到本机的 .env，
哪个测试模块先导入 bot，config 就按那时的环境定下来（load_dotenv 不覆盖已存在的环境变量）。
"""
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
# RAG 库 / live 状态文件 / 谱都写在当前目录下
os.chdir(tempfile.mkdtemp(prefix="lichess-bot-test-"))
os.environ.update({"AUTO_RECALL_K": "0", "SELF_CHECK_ROUNDS": "2", "OPENING_FAST_MOVES": "0",
                   "COMPLEXITY_CHECK": "1", "THINK_LADDER": "default", "MATERIAL_LEAD_SKIP": "12",
                   "BOARD_RELATIONS": "0", "ANALYSIS_BOARD": "0", "PLAN_MEMORY": "1",
                   "HANG_GUARD": "1", "HANG_GUARD_MIN": "2", "HANG_GUARD_ROUNDS": "2",
                   "TRUNCATE_SALVAGE": "summary", "TRUNCATE_REASONING_TAIL": "4000",
                   # 仓库里的技能内容会变，测试默认不加载；需要时显式传 root
                   "SKILLS_DIR": os.path.join(os.getcwd(), "no-skills"), "EXPERIENCE_IN_PLAY": "0"})
SKILLS_ROOT = os.path.join(ROOT, "skills")
