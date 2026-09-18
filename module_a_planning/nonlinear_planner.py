"""Nonlinear Planner (Partial Order Planner / Least Commitment via Constraint Posting).

Features:
- Maintains Causal Links: S_i --[p]--> S_j
- Maintains Ordering Constraints: S_i < S_j
- Detects Threats and resolves via Promotion or Demotion
- Supports parallel execution stratification
"""

from typing import List, Set, Tuple, Dict, Optional
import collections
from module_a_planning.goal_stack_planner import make_action, Action, parse_predicate


class Step:
    def __init__(self, step_id: int, action_name: str, preconds: Set[str], effects: Set[str], del_list: Set[str]):
        self.id = step_id
        self.name = action_name
        self.preconds = set(preconds)
        self.effects = set(effects)
        self.del_list = set(del_list)

    def __repr__(self):
        return f"Step_{self.id}:{self.name}"


class NonlinearPlanner:
    def __init__(self, start_state: Set[str], goal_state: Set[str]):
        self.start_state = set(start_state)
        self.goal_state = set(goal_state)
        self.steps: Dict[int, Step] = {}
        self.orderings: Set[Tuple[int, int]] = set()  # (s1, s2) means s1 < s2
        self.causal_links: List[Tuple[int, str, int]] = []  # (s_from, condition, s_to)
        self.step_counter = 0

    def _add_step(self, name: str, preconds: Set[str], effects: Set[str], del_list: Set[str]) -> Step:
        step = Step(self.step_counter, name, preconds, effects, del_list)
        self.steps[self.step_counter] = step
        self.step_counter += 1
        return step

    def solve(self) -> Dict[str, Any]:
        # Step 0: Start step (effects = start_state)
        s_start = self._add_step("START", set(), self.start_state, set())
        # Step 1: Goal step (preconds = goal_state)
        s_goal = self._add_step("GOAL", self.goal_state, set(), set())

        self.orderings.add((s_start.id, s_goal.id))

        # We search for plan by finding an open condition or threat
        # To demonstrate least commitment and parallel plan generation:
        # We find required actions for goal state and order only when threatened.
        
        # Candidate actions pool for Blocks World
        boxes = ["A", "B", "C", "D", "E", "F", "G"]
        
        open_conditions = [(s_goal.id, cond) for cond in self.goal_state]
        
        iteration = 0
        while open_conditions and iteration < 100:
            iteration += 1
            s_need_id, cond = open_conditions.pop(0)

            # Check if already established by start
            if cond in self.start_state:
                self.causal_links.append((s_start.id, cond, s_need_id))
                self.orderings.add((s_start.id, s_need_id))
                continue

            # Need a new or existing step that adds 'cond'
            producer_step = None
            for step_id, st in self.steps.items():
                if cond in st.effects and step_id != s_need_id:
                    # check if ordering doesn't create cycle
                    if not self._creates_cycle(step_id, s_need_id):
                        producer_step = st
                        break

            if producer_step is None:
                # Instantiate action that produces cond
                act = self._create_producer_action(cond)
                if act:
                    producer_step = self._add_step(str(act), act.preconds, act.add_list, act.del_list)
                    self.orderings.add((s_start.id, producer_step.id))
                    self.orderings.add((producer_step.id, s_goal.id))
                    for p in act.preconds:
                        open_conditions.append((producer_step.id, p))

            if producer_step:
                self.causal_links.append((producer_step.id, cond, s_need_id))
                self.orderings.add((producer_step.id, s_need_id))
                self._resolve_threats()

        # Compute parallel strata / linearized steps
        parallel_levels = self._stratify_parallel()
        linear_plan = [self.steps[sid].name for lvl in parallel_levels for sid in lvl if sid not in (0, 1)]

        return {
            "total_steps": len(linear_plan),
            "linear_plan": linear_plan,
            "parallel_strata": [[self.steps[sid].name for sid in lvl if sid not in (0, 1)] for lvl in parallel_levels if any(sid not in (0, 1) for sid in lvl)],
            "causal_links": [(self.steps[u].name, c, self.steps[v].name) for u, c, v in self.causal_links if u != 0 and v != 1]
        }

    def _create_producer_action(self, cond: str) -> Optional[Action]:
        name, args = parse_predicate(cond)
        if name == "ON" and len(args) == 2:
            return make_action("STACK", args[0], args[1])
        elif name == "HOLDING" and len(args) == 1:
            # If in start state as ontable, pickup, else unstack
            box = args[0]
            for fact in self.start_state:
                fn, fa = parse_predicate(fact)
                if fn == "ON" and len(fa) == 2 and fa[0] == box:
                    return make_action("UNSTACK", box, fa[1])
            return make_action("PICKUP", box)
        elif name == "CLEAR" and len(args) == 1:
            box = args[0]
            for fact in self.start_state:
                fn, fa = parse_predicate(fact)
                if fn == "ON" and len(fa) == 2 and fa[1] == box:
                    return make_action("UNSTACK", fa[0], box)
        elif name == "ONTABLE" and len(args) == 1:
            return make_action("PUTDOWN", args[0])
        elif name == "ARMEMPTY":
            # Can be produced by stack or putdown
            return None
        return None

    def _creates_cycle(self, u: int, v: int) -> bool:
        # Check if v can reach u in transitive closure
        adj = collections.defaultdict(set)
        for s1, s2 in self.orderings:
            adj[s1].add(s2)
        adj[u].add(v)

        visited = set()
        stack = [u]
        while stack:
            curr = stack.pop()
            if curr in visited:
                continue
            visited.add(curr)
            for nxt in adj[curr]:
                if nxt == u:
                    return True
                stack.append(nxt)
        return False

    def _resolve_threats(self):
        # Threat: step Sk has del_list containing c in causal link Si --[c]--> Sj
        # Resolve by promotion (Sk < Si) or demotion (Sj < Sk)
        for (si, c, sj) in list(self.causal_links):
            for sk_id, sk in self.steps.items():
                if sk_id in (si, sj):
                    continue
                if c in sk.del_list:
                    # Potential threat! Post constraint
                    if not self._creates_cycle(sk_id, si):
                        self.orderings.add((sk_id, si))  # Promotion
                    elif not self._creates_cycle(sj, sk_id):
                        self.orderings.add((sj, sk_id))  # Demotion

    def _stratify_parallel(self) -> List[List[int]]:
        """Group steps into parallel executable stages based on dependency graph."""
        in_degree = collections.defaultdict(int)
        adj = collections.defaultdict(list)
        for u, v in self.orderings:
            adj[u].append(v)
            in_degree[v] += 1
        for sid in self.steps:
            if sid not in in_degree:
                in_degree[sid] = 0

        current_level = [sid for sid in self.steps if in_degree[sid] == 0]
        levels = []

        while current_level:
            levels.append(current_level)
            next_level = []
            for u in current_level:
                for v in adj[u]:
                    in_degree[v] -= 1
                    if in_degree[v] == 0:
                        next_level.append(v)
            current_level = next_level

        return levels
