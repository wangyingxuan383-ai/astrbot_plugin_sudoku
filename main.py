import asyncio
import json
import os
import random
import re
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from astrbot.api import logger
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, StarTools, register
from astrbot.core.config.astrbot_config import AstrBotConfig
from astrbot.core.platform.sources.aiocqhttp.aiocqhttp_message_event import (
    AiocqhttpMessageEvent,
)
from astrbot.core.utils.astrbot_path import get_astrbot_data_path

PLUGIN_NAME = "astrbot_plugin_sudoku"

DIFFICULTIES = {
    "easy": {"label": "简单", "min_clues": 36, "max_clues": 45},
    "medium": {"label": "中等", "min_clues": 32, "max_clues": 35},
    "hard": {"label": "困难", "min_clues": 28, "max_clues": 31},
}

DIFFICULTY_ALIASES = {
    "简单": "easy",
    "初级": "easy",
    "easy": "easy",
    "中等": "medium",
    "普通": "medium",
    "medium": "medium",
    "困难": "hard",
    "高级": "hard",
    "hard": "hard",
}

COMMAND_PREFIXES = ["/数独", "数独", "/sudoku", "sudoku"]

try:
    from PIL import Image, ImageDraw, ImageFont

    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False


@dataclass
class SudokuGame:
    puzzle: str
    solution: str
    grid: str
    difficulty: str
    started_at: int
    last_active: int
    lives: int
    contributions: Dict[str, Dict[str, int]]
    names: Dict[str, str]


@register(PLUGIN_NAME, "codex", "数独插件（唯一解/持久化）", "1.0.0")
class SudokuPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.conf = config
        self.data_dir = Path(get_astrbot_data_path()) / "plugin_data" / PLUGIN_NAME
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.games_path = self.data_dir / "games.json"
        self.stats_path = self.data_dir / "stats.json"
        self.render_cache_dir = Path(StarTools.get_data_dir(PLUGIN_NAME)) / "cache"
        self.render_cache_dir.mkdir(parents=True, exist_ok=True)

        self.games_lock = asyncio.Lock()
        self.stats_lock = asyncio.Lock()

        self.games: Dict[str, SudokuGame] = {}
        self.stats: Dict[str, Dict] = {"users": {}}

        self._cleanup_task: Optional[asyncio.Task] = None

        self._last_message_id: Dict[str, int] = {}
        self._renderer = SudokuRenderer(self.conf) if PIL_AVAILABLE else None

        self._load_games()
        self._load_stats()

    async def initialize(self):
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def terminate(self):
        if self._cleanup_task:
            self._cleanup_task.cancel()
        if self.render_cache_dir.exists():
            for item in self.render_cache_dir.glob("*.png"):
                try:
                    item.unlink()
                except Exception:
                    pass
        self.games.clear()

    @filter.command("数独", alias={"sudoku"})
    async def sudoku_command(self, event: AstrMessageEvent):
        event.stop_event()
        text = (event.message_str or "").strip()
        rest = self._strip_command_prefix(text)
        if not rest:
            await self._start_game(event, "medium")
            return

        tokens = rest.split()
        if not tokens:
            await self._start_game(event, "medium")
            return

        sub = tokens[0].lower()
        if sub in ("帮助", "help", "?", "说明"):
            yield event.plain_result(self._help_text())
            return

        if sub in ("查看", "show"):
            await self._show_game(event)
            return

        if sub in ("排行", "排行榜", "rank", "leaderboard"):
            await self._show_leaderboard(event)
            return

        if sub in ("结束", "放弃", "quit", "exit"):
            await self._end_game(event)
            return

        fill_pairs = self._parse_fill_pairs(tokens)
        if fill_pairs:
            await self._apply_fill_pairs(event, fill_pairs)
            return

        if sub in ("填", "填入", "set"):
            if len(tokens) < 3:
                yield event.plain_result("用法：/数独 A1 5 或 #数独 a15 / #数独 a21 b23")
                return
            await self._fill_cell(event, tokens[1], tokens[2])
            return

        difficulty = self._parse_difficulty(tokens[0])
        if difficulty:
            await self._start_game(event, difficulty)
            return

        yield event.plain_result("未识别的指令。输入 /数独 帮助 查看用法。")

    @filter.regex(r"^[#＃]+数独\\b.*")
    async def sudoku_quick_fill(self, event: AstrMessageEvent):
        event.stop_event()
        text = (event.message_str or "").strip()
        rest = re.sub(r"^[#＃]+数独\s*", "", text)
        tokens = rest.split()
        if not tokens:
            return
        fill_pairs = self._parse_fill_pairs(tokens)
        if fill_pairs:
            await self._apply_fill_pairs(event, fill_pairs)

    async def _start_game(self, event: AstrMessageEvent, difficulty: str):
        puzzle = await self._generate_puzzle(difficulty)
        if not puzzle:
            await event.send(event.plain_result("生成数独超时，请稍后重试。"))
            return

        now = int(time.time())
        lives = max(1, int(self.conf.get("lives_default", 3)))
        game = SudokuGame(
            puzzle=puzzle["puzzle"],
            solution=puzzle["solution"],
            grid=puzzle["puzzle"],
            difficulty=difficulty,
            started_at=now,
            last_active=now,
            lives=lives,
            contributions={},
            names={},
        )
        async with self.games_lock:
            self.games[event.unified_msg_origin] = game
            await self._save_games_locked()

        await self._send_game_image(event, game, "新游戏已开始")

    async def _show_game(self, event: AstrMessageEvent):
        game = await self._get_game(event)
        if not game:
            await event.send(event.plain_result("当前没有进行中的数独。使用 /数独 开始。"))
            return
        await self._send_game_image(event, game, "当前进度")

    async def _end_game(self, event: AstrMessageEvent):
        async with self.games_lock:
            game = self.games.get(event.unified_msg_origin)
            if not game:
                await event.send(event.plain_result("当前没有进行中的数独。"))
                return
            self.games.pop(event.unified_msg_origin, None)
            await self._save_games_locked()
        await self._finalize_game(event, game, reason="abort")

    async def _fill_cell(
        self,
        event: AstrMessageEvent,
        cell: str,
        value: str,
        render: bool = True,
    ) -> bool:
        game = await self._get_game(event)
        if not game:
            await event.send(event.plain_result("当前没有进行中的数独。"))
            return False

        idx = self._parse_cell(cell)
        if idx is None:
            await event.send(event.plain_result("格子格式错误，请使用 A1-I9。"))
            return False

        try:
            num = int(value)
        except ValueError:
            await event.send(event.plain_result("请输入 1-9 的数字。"))
            return False

        if num < 1 or num > 9:
            await event.send(event.plain_result("请输入 1-9 的数字。"))
            return False

        if game.puzzle[idx] != "0":
            await event.send(event.plain_result("这是题目给定的格子，不能修改。"))
            return False

        grid = list(game.grid)
        if not self._is_valid_move(grid, idx, str(num)):
            await event.send(event.plain_result("该数字与当前盘面冲突。"))
            return False

        user_id = str(event.get_sender_id())
        user_name = event.get_sender_name()
        game.names[user_id] = user_name
        contrib = game.contributions.setdefault(user_id, {"correct": 0, "wrong": 0, "score": 0})

        if self.conf.get("check_solution_on_fill", True):
            if game.solution[idx] != str(num):
                game.lives -= 1
                contrib["wrong"] += 1
                contrib["score"] -= int(self.conf.get("score_penalty_wrong", 2))
                game.last_active = int(time.time())
                async with self.games_lock:
                    if game.lives <= 0:
                        self.games.pop(event.unified_msg_origin, None)
                        await self._save_games_locked()
                    else:
                        self.games[event.unified_msg_origin] = game
                        await self._save_games_locked()
                if game.lives <= 0:
                    await event.send(event.plain_result("填错次数耗尽，游戏结束。"))
                    await self._finalize_game(event, game, reason="fail")
                else:
                    await event.send(
                        event.plain_result(f"该数字与唯一解不符，生命值 -1（剩余 {game.lives}）")
                    )
                return False

        grid[idx] = str(num)
        contrib["correct"] += 1
        contrib["score"] += int(self.conf.get("score_correct", 1))
        game.grid = "".join(grid)
        game.last_active = int(time.time())
        async with self.games_lock:
            self.games[event.unified_msg_origin] = game
            await self._save_games_locked()

        if "0" not in game.grid and game.grid == game.solution:
            if render:
                await self._send_game_image(event, game, "恭喜完成！")
            async with self.games_lock:
                self.games.pop(event.unified_msg_origin, None)
                await self._save_games_locked()
            await self._finalize_game(event, game, reason="complete")
            return False

        if render:
            await self._send_game_image(event, game, "已更新")
        return True

    def _help_text(self) -> str:
        return (
            "数独指令：\n"
            "- /数独 [简单/中等/困难] 开始新游戏\n"
            "- /数独 查看\n"
            "- /数独 A1 5 或 /数独 A11 (在A1填5)\n"
            "- #数独 a15 或 #数独 a21 b23 (批量填)\n"
            "- /数独 排行\n"
            "- /数独 结束\n"
        )

    async def _get_game(self, event: AstrMessageEvent) -> Optional[SudokuGame]:
        async with self.games_lock:
            return self.games.get(event.unified_msg_origin)

    def _strip_command_prefix(self, text: str) -> str:
        for prefix in COMMAND_PREFIXES:
            if text.startswith(prefix):
                return text[len(prefix) :].strip()
        return text

    def _parse_difficulty(self, token: str) -> Optional[str]:
        if not token:
            return None
        norm = token.lower()
        return DIFFICULTY_ALIASES.get(token) or DIFFICULTY_ALIASES.get(norm)

    def _parse_cell(self, cell: str) -> Optional[int]:
        cell = cell.strip().upper()
        if len(cell) != 2:
            return None
        col_char, row_char = cell[0], cell[1]
        if col_char < "A" or col_char > "I":
            return None
        if row_char < "1" or row_char > "9":
            return None
        col = ord(col_char) - ord("A")
        row = int(row_char) - 1
        return row * 9 + col

    def _parse_quick_fill_token(self, token: str) -> Optional[Tuple[str, str]]:
        token = token.strip()
        match = re.match(r"^([A-Ia-i])([1-9])([1-9])$", token)
        if not match:
            return None
        cell = f"{match.group(1)}{match.group(2)}"
        value = match.group(3)
        return cell, value

    def _parse_fill_pairs(self, tokens: List[str]) -> Optional[List[Tuple[str, str]]]:
        pairs: List[Tuple[str, str]] = []
        i = 0
        while i < len(tokens):
            token = tokens[i]
            compact = self._parse_quick_fill_token(token)
            if compact:
                pairs.append(compact)
                i += 1
                continue
            if re.match(r"^[A-Ia-i][1-9]$", token):
                if i + 1 >= len(tokens):
                    break
                value = tokens[i + 1]
                if not re.match(r"^[1-9]$", value):
                    break
                pairs.append((token, value))
                i += 2
                continue
            break
        return pairs if pairs else None

    async def _apply_fill_pairs(self, event: AstrMessageEvent, pairs: List[Tuple[str, str]]):
        for idx, (cell, value) in enumerate(pairs):
            is_last = idx == len(pairs) - 1
            ok = await self._fill_cell(event, cell, value, render=is_last)
            if not ok:
                break


    async def _cleanup_loop(self):
        while True:
            try:
                await asyncio.sleep(max(1, int(self.conf.get("cleanup_interval_minutes", 10))) * 60)
                await self._cleanup_games()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(f"数独清理任务异常: {exc}")

    async def _cleanup_games(self):
        ttl_minutes = int(self.conf.get("game_ttl_minutes", 120))
        cutoff = int(time.time()) - ttl_minutes * 60
        async with self.games_lock:
            stale = [k for k, v in self.games.items() if v.last_active < cutoff]
            for key in stale:
                self.games.pop(key, None)
            if stale:
                await self._save_games_locked()



    def _load_games(self):
        if not self.games_path.exists():
            return
        try:
            with self.games_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return
            for key, raw in data.items():
                if not isinstance(raw, dict):
                    continue
                game = SudokuGame(
                    puzzle=raw.get("puzzle", ""),
                    solution=raw.get("solution", ""),
                    grid=raw.get("grid", ""),
                    difficulty=raw.get("difficulty", "medium"),
                    started_at=int(raw.get("started_at", 0)),
                    last_active=int(raw.get("last_active", 0)),
                    lives=int(raw.get("lives", self.conf.get("lives_default", 3))),
                    contributions=raw.get("contributions", {}) or {},
                    names=raw.get("names", {}) or {},
                )
                if len(game.puzzle) == 81 and len(game.solution) == 81 and len(game.grid) == 81:
                    self.games[key] = game
        except Exception as exc:
            logger.error(f"读取数独游戏失败: {exc}")

    async def _save_games_locked(self):
        data = {
            k: {
                "puzzle": v.puzzle,
                "solution": v.solution,
                "grid": v.grid,
                "difficulty": v.difficulty,
                "started_at": v.started_at,
                "last_active": v.last_active,
                "lives": v.lives,
                "contributions": v.contributions,
                "names": v.names,
            }
            for k, v in self.games.items()
        }
        await self._write_json_atomic(self.games_path, data)

    def _load_stats(self):
        if not self.stats_path.exists():
            return
        try:
            with self.stats_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                self.stats = data
        except Exception as exc:
            logger.error(f"读取数独统计失败: {exc}")

    async def _save_stats(self):
        async with self.stats_lock:
            await self._write_json_atomic(self.stats_path, self.stats)

    def _get_user_stat(self, user_id: str, name: str) -> Dict[str, any]:
        users = self.stats.setdefault("users", {})
        stat = users.get(user_id)
        if not stat:
            stat = {
                "name": name,
                "score": 0,
                "wins": {"easy": 0, "medium": 0, "hard": 0},
                "total_time": 0,
                "errors": 0,
                "games": 0,
            }
            users[user_id] = stat
        stat["name"] = name or stat.get("name", user_id)
        return stat

    @staticmethod
    def _format_duration(seconds: int) -> str:
        minutes, sec = divmod(max(0, seconds), 60)
        return f"{minutes:02d}:{sec:02d}"

    async def _finalize_game(self, event: AstrMessageEvent, game: SudokuGame, reason: str):
        now = int(time.time())
        elapsed = self._format_duration(now - game.started_at)
        contributions = game.contributions or {}
        names = game.names or {}

        def display_name(uid: str) -> str:
            return names.get(uid) or uid

        top_user = None
        top_correct = -1
        for uid, stat in contributions.items():
            correct = int(stat.get("correct", 0))
            if correct > top_correct:
                top_correct = correct
                top_user = uid

        wrong_list = [
            f"{display_name(uid)}({stat.get('wrong', 0)})"
            for uid, stat in contributions.items()
            if int(stat.get("wrong", 0)) > 0
        ]
        wrong_text = "、".join(wrong_list) if wrong_list else "无"

        # Update stats
        if contributions:
            for uid, stat in contributions.items():
                user_name = display_name(uid)
                user_stat = self._get_user_stat(uid, user_name)
                user_stat["score"] += int(stat.get("score", 0))
                user_stat["errors"] += int(stat.get("wrong", 0))
                user_stat["games"] += 1

        if reason == "complete" and top_user:
            bonus = int(self.conf.get(f"score_bonus_{game.difficulty}", 10))
            winner_stat = self._get_user_stat(top_user, display_name(top_user))
            winner_stat["score"] += bonus
            winner_stat["wins"][game.difficulty] = winner_stat["wins"].get(game.difficulty, 0) + 1
            winner_stat["total_time"] += now - game.started_at

        await self._save_stats()

        participants_lines = []
        for uid, stat in contributions.items():
            participants_lines.append(
                f"- {display_name(uid)} 正确{stat.get('correct',0)} 错误{stat.get('wrong',0)} 分数{stat.get('score',0)}"
            )
        participants_text = "\n".join(participants_lines) if participants_lines else "无"

        top_text = (
            f"{display_name(top_user)}（正确 {top_correct} 次）" if top_user else "无"
        )
        result_label = "完成" if reason == "complete" else "结束"
        if reason == "fail":
            result_label = "失败"

        summary = (
            f"本局{result_label}（{DIFFICULTIES[game.difficulty]['label']}）\n"
            f"用时：{elapsed}\n"
            f"贡献最多：{top_text}\n"
            f"扣命出错：{wrong_text}\n"
            f"参与记录：\n{participants_text}"
        )
        await event.send(event.plain_result(summary))

    async def _show_leaderboard(self, event: AstrMessageEvent):
        users = self.stats.get("users", {})
        if not users:
            await event.send(event.plain_result("暂无排行榜数据。"))
            return

        entries = []
        for uid, stat in users.items():
            wins = stat.get("wins", {})
            total_wins = sum(int(v) for v in wins.values())
            total_time = int(stat.get("total_time", 0))
            avg_time = total_time // total_wins if total_wins > 0 else 999999
            entries.append(
                (
                    int(stat.get("score", 0)),
                    avg_time,
                    uid,
                    stat,
                )
            )
        entries.sort(key=lambda x: (-x[0], x[1]))

        lines = ["数独排行榜（按分数/平均用时）："]
        for idx, (score, avg_time, uid, stat) in enumerate(entries[:10], start=1):
            wins = stat.get("wins", {})
            total_wins = sum(int(v) for v in wins.values())
            avg_time_str = self._format_duration(avg_time) if total_wins > 0 else "--:--"
            lines.append(
                f"{idx}. {stat.get('name', uid)} 分数{score} 平均用时{avg_time_str} "
                f"胜场(简{wins.get('easy',0)}/中{wins.get('medium',0)}/难{wins.get('hard',0)})"
            )
        await event.send(event.plain_result("\n".join(lines)))

    async def _write_json_atomic(self, path: Path, data: dict):
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        try:
            with tmp_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
            os.replace(tmp_path, path)
        except Exception as exc:
            logger.error(f"写入文件失败: {path} {exc}")

    async def _generate_puzzle(self, difficulty: str) -> Optional[dict]:
        timeout = int(self.conf.get("generation_timeout_seconds", 5))
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(self._generate_puzzle_sync, difficulty, timeout),
                timeout=timeout + 1,
            )
        except asyncio.TimeoutError:
            return None
        except Exception as exc:
            logger.error(f"生成数独异常: {exc}")
            return None

    def _generate_puzzle_sync(self, difficulty: str, timeout: int) -> Optional[dict]:
        deadline = time.monotonic() + max(1, timeout)
        while time.monotonic() < deadline:
            solution = self._generate_solution(deadline)
            if solution is None:
                continue
            if not self._is_solution_valid(solution):
                continue
            puzzle = self._dig_holes(solution, difficulty, deadline)
            if puzzle is None:
                continue
            return {
                "puzzle": puzzle,
                "solution": solution,
                "difficulty": difficulty,
                "created_at": int(time.time()),
            }
        return None

    def _generate_solution(self, deadline: float) -> Optional[str]:
        grid = ["0"] * 81

        def backtrack() -> bool:
            if time.monotonic() > deadline:
                return False
            idx = self._find_empty_cell(grid)
            if idx is None:
                return True
            candidates = self._candidates(grid, idx)
            random.shuffle(candidates)
            for num in candidates:
                grid[idx] = str(num)
                if backtrack():
                    return True
                grid[idx] = "0"
            return False

        if backtrack():
            return "".join(grid)
        return None

    def _dig_holes(self, solution: str, difficulty: str, deadline: float) -> Optional[str]:
        info = DIFFICULTIES[difficulty]
        target = random.randint(info["min_clues"], info["max_clues"])
        puzzle = list(solution)
        indices = list(range(81))
        random.shuffle(indices)

        for idx in indices:
            if time.monotonic() > deadline:
                return None
            if puzzle.count("0") >= 81 - target:
                break
            backup = puzzle[idx]
            puzzle[idx] = "0"
            if self._count_solutions(puzzle, 2, deadline) != 1:
                puzzle[idx] = backup
        clues = 81 - puzzle.count("0")
        if clues < info["min_clues"] or clues > info["max_clues"]:
            return None
        if self._count_solutions(puzzle, 2, deadline) != 1:
            return None
        return "".join(puzzle)

    def _count_solutions(self, grid: List[str], limit: int, deadline: float) -> int:
        if time.monotonic() > deadline:
            return 0
        idx = self._find_empty_cell(grid)
        if idx is None:
            return 1
        candidates = self._candidates(grid, idx)
        if not candidates:
            return 0
        total = 0
        for num in candidates:
            grid[idx] = str(num)
            total += self._count_solutions(grid, limit, deadline)
            if total >= limit:
                grid[idx] = "0"
                return total
            grid[idx] = "0"
        return total

    def _find_empty_cell(self, grid: List[str]) -> Optional[int]:
        min_count = 10
        min_idx: Optional[int] = None
        for idx, val in enumerate(grid):
            if val != "0":
                continue
            candidates = self._candidates(grid, idx)
            if not candidates:
                return idx
            if len(candidates) < min_count:
                min_count = len(candidates)
                min_idx = idx
            if min_count == 1:
                break
        return min_idx

    def _candidates(self, grid: List[str], idx: int) -> List[int]:
        row = idx // 9
        col = idx % 9
        used = set()
        for i in range(9):
            used.add(grid[row * 9 + i])
            used.add(grid[i * 9 + col])
        box_row = (row // 3) * 3
        box_col = (col // 3) * 3
        for r in range(box_row, box_row + 3):
            for c in range(box_col, box_col + 3):
                used.add(grid[r * 9 + c])
        return [n for n in range(1, 10) if str(n) not in used]

    def _is_valid_move(self, grid: List[str], idx: int, value: str) -> bool:
        row = idx // 9
        col = idx % 9
        for i in range(9):
            if i != col and grid[row * 9 + i] == value:
                return False
            if i != row and grid[i * 9 + col] == value:
                return False
        box_row = (row // 3) * 3
        box_col = (col // 3) * 3
        for r in range(box_row, box_row + 3):
            for c in range(box_col, box_col + 3):
                pos = r * 9 + c
                if pos != idx and grid[pos] == value:
                    return False
        return True

    def _is_solution_valid(self, solution: str) -> bool:
        if len(solution) != 81:
            return False
        grid = list(solution)
        digits = {str(i) for i in range(1, 10)}
        for r in range(9):
            row_vals = set(grid[r * 9 : r * 9 + 9])
            if row_vals != digits:
                return False
        for c in range(9):
            col_vals = {grid[r * 9 + c] for r in range(9)}
            if col_vals != digits:
                return False
        for br in range(0, 9, 3):
            for bc in range(0, 9, 3):
                box_vals = set()
                for r in range(br, br + 3):
                    for c in range(bc, bc + 3):
                        box_vals.add(grid[r * 9 + c])
                if box_vals != digits:
                    return False
        return True

    async def _send_game_image(self, event: AstrMessageEvent, game: SudokuGame, title: str):
        if self._renderer and self.conf.get("use_pil_renderer", True):
            try:
                img_bytes = self._renderer.render(game=game, title=title)
                img_path = self._save_img_bytes(event, img_bytes)
                await self._send_image(event, img_path)
                return
            except Exception as exc:
                logger.error(f"数独 PIL 渲染失败: {exc}")

        try:
            url = await self._render_html_board(game, title)
            await self._send_image(event, url)
        except Exception as exc:
            logger.error(f"渲染数独失败: {exc}")
            await event.send(event.plain_result("渲染失败，请检查渲染环境。"))

    def _save_img_bytes(self, event: AstrMessageEvent, img_bytes: bytes) -> str:
        sid = getattr(event, "session_id", "session")
        uid = event.get_sender_id()
        fname = f"sudoku_{sid}_{uid}.png"
        path = self.render_cache_dir / fname
        path.write_bytes(img_bytes)
        return str(path.absolute())

    async def _send_image(self, event: AstrMessageEvent, image_ref: str):
        replace = bool(self.conf.get("replace_last_image", True))
        if not replace or not isinstance(event, AiocqhttpMessageEvent):
            await event.send(event.image_result(image_ref))
            return

        key = f"{event.session_id}:{event.get_sender_id()}"
        payloads = {"message": [{"type": "image", "data": {"file": image_ref}}]}
        try:
            if event.is_private_chat():
                payloads["user_id"] = event.get_sender_id()
                result = await event.bot.api.call_action("send_private_msg", **payloads)
            else:
                payloads["group_id"] = event.get_group_id()
                result = await event.bot.api.call_action("send_group_msg", **payloads)
            message_id = result.get("message_id") if isinstance(result, dict) else None
        except Exception:
            await event.send(event.image_result(image_ref))
            return

        last_id = self._last_message_id.get(key)
        if last_id:
            try:
                await event.bot.delete_msg(message_id=last_id)
            except Exception:
                pass
        if message_id:
            self._last_message_id[key] = message_id

    async def _render_html_board(self, game: SudokuGame, title: str) -> str:
        cells = []
        filled = 0
        for r in range(9):
            row_cells = []
            for c in range(9):
                idx = r * 9 + c
                value = game.grid[idx]
                given = game.puzzle[idx] != "0"
                if value != "0":
                    filled += 1
                classes = []
                classes.append("given" if given else "user")
                if value == "0":
                    classes.append("empty")
                if c in (2, 5):
                    classes.append("br")
                if r in (2, 5):
                    classes.append("bb")
                row_cells.append(
                    {
                        "value": "" if value == "0" else value,
                        "class": " ".join(classes + ([f"n{value}"] if value != "0" else [])),
                    }
                )
            cells.append(row_cells)

        cell_size = int(self.conf.get("image_cell_size", 48))
        font_size = max(14, int(cell_size * 0.45))
        progress = f"{filled}/81"
        data = {
            "cells": cells,
            "difficulty": DIFFICULTIES[game.difficulty]["label"],
            "title": title,
            "cell_size": cell_size,
            "font_size": font_size,
            "progress": progress,
            "lives": game.lives,
            "tip": "#数独 a11填数",
        }

        html = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<style>
body { margin: 0; font-family: "Noto Sans", "Microsoft YaHei", sans-serif; }
.container { padding: 14px 18px; background: linear-gradient(180deg, #f7f3ea 0%, #e7dccb 100%); }
.header { display: flex; align-items: baseline; justify-content: space-between; margin-bottom: 2px; }
.subheader { text-align: center; font-size: 13px; color: #666; margin-bottom: 8px; }
.title { font-size: 20px; font-weight: 700; color: #222; }
.meta { font-size: 14px; color: #555; }
.card { display: inline-block; padding: 10px; background: #fcfbf7; border: 1px solid #d1c6b8; border-radius: 12px; box-shadow: 0 6px 12px rgba(0,0,0,0.12); }
.sudoku { border-collapse: collapse; }
.sudoku th, .sudoku td {
  width: {{ cell_size }}px;
  height: {{ cell_size }}px;
  text-align: center;
  vertical-align: middle;
  font-size: {{ font_size }}px;
  border: 1px solid #333;
}
.sudoku th { background: #eee7dc; font-weight: 600; }
.sudoku td.given { color: #111; font-weight: 700; }
.sudoku td.user { color: #1a5fd1; font-weight: 600; }
.sudoku td.empty { color: #aaa; }
.sudoku td.n1 { color: #1f77b4; font-weight: 700; }
.sudoku td.n2 { color: #2ca02c; font-weight: 700; }
.sudoku td.n3 { color: #d62728; font-weight: 700; }
.sudoku td.n4 { color: #9467bd; font-weight: 700; }
.sudoku td.n5 { color: #8c564b; font-weight: 700; }
.sudoku td.n6 { color: #17becf; font-weight: 700; }
.sudoku td.n7 { color: #111111; font-weight: 700; }
.sudoku td.n8 { color: #7f7f7f; font-weight: 700; }
.sudoku td.n9 { color: #ff7f0e; font-weight: 700; }
.sudoku td.br { border-right: 3px solid #111; }
.sudoku td.bb { border-bottom: 3px solid #111; }
.sudoku th.br { border-right: 3px solid #111; }
.sudoku th.bb { border-bottom: 3px solid #111; }
</style>
</head>
<body>
  <div class="container">
    <div class="header">
      <div class="title">{{ title }} · {{ difficulty }}</div>
      <div class="meta">进度 {{ progress }} · 生命 {{ lives }}</div>
    </div>
    <div class="subheader">{{ tip }}</div>
    <div class="card">
      <table class="sudoku">
        <tr>
          <th></th>
          {% for c in range(9) %}
          <th class="{% if c in (2,5) %}br{% endif %}">{{ "ABCDEFGHI"[c] }}</th>
          {% endfor %}
        </tr>
        {% for r in range(9) %}
        <tr>
          <th class="{% if r in (2,5) %}bb{% endif %}">{{ r + 1 }}</th>
          {% for c in range(9) %}
          {% set cell = cells[r][c] %}
          <td class="{{ cell.class }}">{{ cell.value }}</td>
          {% endfor %}
        </tr>
        {% endfor %}
      </table>
    </div>
  </div>
</body>
</html>
"""
        return await self.html_render(html, data, options={"type": "png", "full_page": True})


if PIL_AVAILABLE:
    class SudokuRenderer:
        def __init__(self, config: AstrBotConfig):
            self.cell_size = int(config.get("image_cell_size", 48))
            self.padding = max(8, int(self.cell_size * 0.25))
            self.label_size = max(12, int(self.cell_size * 0.45))
            self.header_height = max(40, int(self.cell_size * 1.2))
            self.font_path = self._resolve_font_path(config)
            self.font_cell = self._load_font(int(self.cell_size * 0.6))
            self.font_label = self._load_font(int(self.cell_size * 0.35))
            self.font_header = self._load_font(int(self.cell_size * 0.45))
            self.number_colors = {
                "1": "#1f77b4",
                "2": "#2ca02c",
                "3": "#d62728",
                "4": "#9467bd",
                "5": "#8c564b",
                "6": "#17becf",
                "7": "#111111",
                "8": "#7f7f7f",
                "9": "#ff7f0e",
            }

        def _resolve_font_path(self, config: AstrBotConfig) -> str:
            raw = str(config.get("font_path", "")).strip()
            candidates: List[Path] = []
            if raw:
                candidates.append(Path(raw))
                if not raw.startswith("/"):
                    candidates.append(Path(__file__).parent / raw)
            candidates.append(Path(__file__).parent / "assets" / "LXGWWenKai-Regular.ttf")
            for path in candidates:
                if path.exists():
                    return str(path)
            return ""

        def _load_font(self, size: int) -> ImageFont.ImageFont:
            try:
                if self.font_path:
                    return ImageFont.truetype(font=self.font_path, size=size)
            except Exception:
                pass
            return ImageFont.load_default()

        def render(self, game: SudokuGame, title: str) -> bytes:
            board = self._build_board(game, title)
            output = BytesIO()
            board.save(output, format="PNG")
            output.seek(0)
            return output.getvalue()

        def _build_board(self, game: SudokuGame, title: str) -> Image.Image:
            cell = self.cell_size
            label = self.label_size
            pad = self.padding
            header = self.header_height

            width = pad * 2 + label + cell * 9
            height = pad * 2 + header + label + cell * 9

            img = self._create_background(width, height)
            draw = ImageDraw.Draw(img)

            progress = f"{81 - game.grid.count('0')}/81"
            header_text = f"{title} · {DIFFICULTIES[game.difficulty]['label']}"
            tip_text = "#数独 a11填数"
            line1_y = pad
            line2_y = pad + self.font_header.size + 4
            draw.text((pad, line1_y), header_text, font=self.font_header, fill="#2a2a2a")
            draw.text(
                (width - pad - 140, line1_y),
                f"进度 {progress} 生命 {game.lives}",
                font=self.font_label,
                fill="#555",
            )
            self._draw_centered(
                draw,
                tip_text,
                (width / 2, line2_y + self.font_label.size / 2),
                self.font_label,
                "#666",
            )

            board_x = pad + label
            board_y = pad + header + label

            card_pad = max(8, int(cell * 0.2))
            card_rect = (
                board_x - card_pad,
                board_y - card_pad,
                board_x + cell * 9 + card_pad,
                board_y + cell * 9 + card_pad,
            )
            img = self._draw_card(img, card_rect)
            draw = ImageDraw.Draw(img)

            for i in range(9):
                col_char = chr(ord("A") + i)
                x = board_x + i * cell + cell / 2
                y = pad + header + label / 2
                self._draw_centered(draw, col_char, (x, y), self.font_label, "#3a3a3a")

                row_char = str(i + 1)
                x = pad + label / 2
                y = board_y + i * cell + cell / 2
                self._draw_centered(draw, row_char, (x, y), self.font_label, "#3a3a3a")

            for i in range(10):
                thickness = 3 if i % 3 == 0 else 1
                x = board_x + i * cell
                draw.line([(x, board_y), (x, board_y + cell * 9)], fill="#1a1a1a", width=thickness)
                y = board_y + i * cell
                draw.line([(board_x, y), (board_x + cell * 9, y)], fill="#1a1a1a", width=thickness)

            for r in range(9):
                for c in range(9):
                    idx = r * 9 + c
                    val = game.grid[idx]
                    if val == "0":
                        continue
                    color = self.number_colors.get(val, "#111")
                    x = board_x + c * cell + cell / 2
                    y = board_y + r * cell + cell / 2
                    self._draw_bold_centered(draw, val, (x, y), self.font_cell, color)

            return img

        def _create_background(self, width: int, height: int) -> Image.Image:
            top = Image.new("RGB", (width, height), "#f7f3ea")
            bottom = Image.new("RGB", (width, height), "#e7dccb")
            try:
                gradient = Image.linear_gradient("L").resize((width, height))
                base = Image.composite(bottom, top, gradient)
            except Exception:
                base = top
            return base

        def _draw_card(self, img: Image.Image, rect: tuple[int, int, int, int]) -> Image.Image:
            x1, y1, x2, y2 = rect
            radius = max(8, int(self.cell_size * 0.2))
            shadow = Image.new("RGBA", img.size, (0, 0, 0, 0))
            shadow_draw = ImageDraw.Draw(shadow)
            try:
                shadow_draw.rounded_rectangle(
                    (x1 + 4, y1 + 4, x2 + 4, y2 + 4),
                    radius=radius,
                    fill=(0, 0, 0, 50),
                )
            except Exception:
                shadow_draw.rectangle((x1 + 4, y1 + 4, x2 + 4, y2 + 4), fill=(0, 0, 0, 50))

            composed = Image.alpha_composite(img.convert("RGBA"), shadow)
            draw = ImageDraw.Draw(composed)
            try:
                draw.rounded_rectangle(
                    rect,
                    radius=radius,
                    fill="#fcfbf7",
                    outline="#d1c6b8",
                    width=2,
                )
            except Exception:
                draw.rectangle(rect, fill="#fcfbf7", outline="#d1c6b8", width=2)
            return composed.convert("RGB")

        @staticmethod
        def _draw_centered(
            draw: ImageDraw.ImageDraw,
            text: str,
            center: tuple[float, float],
            font: ImageFont.ImageFont,
            fill: str,
        ):
            bbox = draw.textbbox((0, 0), text, font=font)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            x = center[0] - w / 2
            y = center[1] - h / 2
            draw.text((x, y), text, font=font, fill=fill)

        def _draw_bold_centered(
            self,
            draw: ImageDraw.ImageDraw,
            text: str,
            center: tuple[float, float],
            font: ImageFont.ImageFont,
            fill: str,
        ):
            self._draw_centered(draw, text, (center[0] + 0.6, center[1]), font, fill)
            self._draw_centered(draw, text, (center[0], center[1] + 0.6), font, fill)
else:
    SudokuRenderer = None  # type: ignore
