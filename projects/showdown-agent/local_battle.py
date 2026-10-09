#!/usr/bin/env python3
"""
Interactive Terminal 1v1 Battle vs Amnesia-AI
=============================================
Play directly against the strongest Amnesia-AI RL/PPO model (gen8ou_ppo.pt)
in an interactive command-line battle.
"""

import os
import sys
import json
import random
import torch
import numpy as np

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../behavioral-cloning")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../reinforcement-learning")))

from mechanics import (
    ShowdownData, clean_id, estimate_damage_pct, get_type_multiplier,
    get_hazard_damage_pct, TYPES
)
from parser import BattlePokemon, PlayerState, build_state_vector, build_candidate_actions
from continuous_gen8ou_trainer import Gen8FastSimEnvironment, Gen8PPOActorCritic


def print_hp_bar(pct: float, length: int = 20) -> str:
    filled = int(round((pct / 100.0) * length))
    filled = max(0, min(length, filled))
    bar = "█" * filled + "░" * (length - filled)
    return f"[{bar}] {pct:.1f}%"


def run_local_battle(checkpoint_path: str = "projects/reinforcement-learning/weights/gen8ou_ppo.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = ShowdownData.get()

    teams_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/gen8ou_teams.json"))
    env = Gen8FastSimEnvironment(teams_file)

    # Load Model
    model = Gen8PPOActorCritic().to(device)
    if os.path.exists(checkpoint_path):
        st = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(st.get("model_state_dict", st), strict=False)
        print(f"✅ Amnesia-AI carregado com sucesso ({checkpoint_path}) no dispositivo {device}!")
    else:
        print(f"⚠️ Checkpoint {checkpoint_path} não encontrado, usando inicialização padrão.")
    model.eval()

    # Setup Teams
    player_team = env.get_random_team()
    ai_team = env.get_random_team()

    p_active = 0
    ai_active = 0
    p_hazards = {"stealthrock": 0, "spikes": 0, "toxicspikes": 0, "stickyweb": 0}
    ai_hazards = {"stealthrock": 0, "spikes": 0, "toxicspikes": 0, "stickyweb": 0}

    print("\n" + "=" * 65)
    print("⚔️  AMNESIA-AI vs JOGADOR — DUELO 1v1 GEN 8 OU  ⚔️")
    print("=" * 65)
    print("Seu time:", ", ".join(p["species"] for p in player_team))
    print("Time Amnesia-AI:", ", ".join(p["species"] for p in ai_team))
    print("=" * 65 + "\n")

    turn = 0

    while turn < 80:
        turn += 1
        p_alive = [i for i, p in enumerate(player_team) if p["hp"] > 0]
        ai_alive = [i for i, p in enumerate(ai_team) if p["hp"] > 0]

        if not p_alive:
            print("\n💀 Todos os seus Pokémon foram nocauteados! Amnesia-AI venceu a partida!")
            break
        if not ai_alive:
            print("\n🏆 Parabéns! Você derrotou todos os Pokémon de Amnesia-AI e venceu!")
            break

        # Check active fainting
        if player_team[p_active]["hp"] <= 0:
            print(f"\n⚠️ Seu {player_team[p_active]['species']} desmaiou! Escolha um substituto:")
            for idx, p_idx in enumerate(p_alive):
                print(f"  [{idx + 1}] {player_team[p_idx]['species']} ({player_team[p_idx]['hp']:.1f}% HP)")
            while True:
                choice = input("Substituto número: ").strip()
                if choice.isdigit() and 1 <= int(choice) <= len(p_alive):
                    p_active = p_alive[int(choice) - 1]
                    h_dmg = get_hazard_damage_pct(player_team[p_active]["species"], p_hazards, player_team[p_active].get("item"))
                    player_team[p_active]["hp"] = max(0.0, player_team[p_active]["hp"] - h_dmg * 100.0)
                    print(f"➡️ Você enviou {player_team[p_active]['species']}!")
                    if h_dmg > 0:
                        print(f"   (Tomou {h_dmg*100:.1f}% de dano de hazards na entrada)")
                    break

        if ai_team[ai_active]["hp"] <= 0:
            ai_active = ai_alive[0]
            h_dmg = get_hazard_damage_pct(ai_team[ai_active]["species"], ai_hazards, ai_team[ai_active].get("item"))
            ai_team[ai_active]["hp"] = max(0.0, ai_team[ai_active]["hp"] - h_dmg * 100.0)
            print(f"\n🤖 Amnesia-AI enviou {ai_team[ai_active]['species']} ({ai_team[ai_active]['hp']:.1f}% HP)!")

        # Turn Header
        print("\n" + "-" * 65)
        print(f"📍 TURNO {turn}")
        print(f"🤖 [Amnesia-AI]: {ai_team[ai_active]['species']:<15} {print_hp_bar(ai_team[ai_active]['hp'])}")
        print(f"👤 [Você]:       {player_team[p_active]['species']:<15} {print_hp_bar(player_team[p_active]['hp'])}")
        print("-" * 65)

        # Show player choices
        p_moves = player_team[p_active]["moves"]
        print("Golpes:")
        for m_idx, m_name in enumerate(p_moves):
            est_dmg = estimate_damage_pct(player_team[p_active]["species"], ai_team[ai_active]["species"], m_name, {}, {}) * 100.0
            print(f"  [{m_idx + 1}] {m_name:<18} (Dano estimado: ~{est_dmg:.1f}%)")

        bench_switches = [i for i in range(len(player_team)) if player_team[i]["hp"] > 0 and i != p_active]
        print("Trocas:")
        switch_map = {}
        for s_idx, b_idx in enumerate(bench_switches):
            slot_num = len(p_moves) + s_idx + 1
            switch_map[slot_num] = b_idx
            print(f"  [{slot_num}] Trocar para {player_team[b_idx]['species']} ({player_team[b_idx]['hp']:.1f}% HP)")

        # Get Player Action
        player_act = 0
        is_switch = False
        while True:
            raw_input = input(f"Sua ação (1-{len(p_moves) + len(bench_switches)}): ").strip()
            if raw_input.isdigit():
                val = int(raw_input)
                if 1 <= val <= len(p_moves):
                    player_act = val - 1
                    is_switch = False
                    break
                elif val in switch_map:
                    player_act = switch_map[val]
                    is_switch = True
                    break
            print("❌ Opção inválida. Digite o número correspondente.")

        # AI Action via Neural Network
        s_ai, a_ai, mask_ai = env.encode_state(ai_team, player_team, ai_active, p_active, ai_hazards, p_hazards)
        with torch.no_grad():
            s_t = torch.tensor(s_ai, dtype=torch.float32, device=device).unsqueeze(0)
            a_t = torch.tensor(a_ai, dtype=torch.float32, device=device).unsqueeze(0)
            m_t = torch.tensor(mask_ai, dtype=torch.float32, device=device).unsqueeze(0)
            logits_ai, _ = model(s_t, a_t, m_t)
            ai_chosen_slot = torch.argmax(logits_ai, dim=-1).item()

        ai_is_switch = ai_chosen_slot >= 4
        ai_switch_idx = None
        if ai_is_switch:
            ai_sw_slot = ai_chosen_slot - 4
            ai_bench = [i for i, p in enumerate(ai_team) if p["hp"] > 0 and i != ai_active]
            if ai_sw_slot < len(ai_bench):
                ai_switch_idx = ai_bench[ai_sw_slot]
            else:
                ai_is_switch = False
                ai_chosen_slot = 0

        # Simultaneous Execution
        print("\n--- ACONTECIMENTOS DO TURNO ---")
        if is_switch:
            p_active = player_act
            h_dmg = get_hazard_damage_pct(player_team[p_active]["species"], p_hazards, player_team[p_active].get("item"))
            player_team[p_active]["hp"] = max(0.0, player_team[p_active]["hp"] - h_dmg * 100.0)
            print(f"👤 Você trocou para {player_team[p_active]['species']}!")
            if h_dmg > 0:
                print(f"   (Tomou {h_dmg*100:.1f}% de dano residual de hazards)")

        if ai_is_switch and ai_switch_idx is not None:
            ai_active = ai_switch_idx
            h_dmg = get_hazard_damage_pct(ai_team[ai_active]["species"], ai_hazards, ai_team[ai_active].get("item"))
            ai_team[ai_active]["hp"] = max(0.0, ai_team[ai_active]["hp"] - h_dmg * 100.0)
            print(f"🤖 Amnesia-AI trocou para {ai_team[ai_active]['species']}!")
            if h_dmg > 0:
                print(f"   (Tomou {h_dmg*100:.1f}% de dano residual de hazards)")

        # Hazard Moves
        if not is_switch:
            m_p_clean = clean_id(player_team[p_active]["moves"][player_act])
            if m_p_clean == "stealthrock": ai_hazards["stealthrock"] = 1; print("👤 Você colocou Stealth Rock em campo!")
            elif m_p_clean == "spikes": ai_hazards["spikes"] = min(3, ai_hazards["spikes"] + 1); print("👤 Você colocou Spikes em campo!")

        if not ai_is_switch and ai_chosen_slot < len(ai_team[ai_active]["moves"]):
            m_ai_clean = clean_id(ai_team[ai_active]["moves"][ai_chosen_slot])
            if m_ai_clean == "stealthrock": p_hazards["stealthrock"] = 1; print("🤖 Amnesia-AI colocou Stealth Rock em campo!")
            elif m_ai_clean == "spikes": p_hazards["spikes"] = min(3, p_hazards["spikes"] + 1); print("🤖 Amnesia-AI colocou Spikes em campo!")

        # Speed check for attacks
        p_spe = (data.get_pokemon(player_team[p_active]["species"]) or {}).get("baseStats", {}).get("spe", 80)
        ai_spe = (data.get_pokemon(ai_team[ai_active]["species"]) or {}).get("baseStats", {}).get("spe", 80)
        p_faster = p_spe >= ai_spe

        def player_attack():
            if is_switch or player_team[p_active]["hp"] <= 0: return
            move_name = player_team[p_active]["moves"][player_act]
            dmg = estimate_damage_pct(player_team[p_active]["species"], ai_team[ai_active]["species"], move_name, {}, {}) * 100.0
            # Apply slight RNG roll (85% to 100%)
            roll = random.uniform(0.85, 1.0)
            dmg = dmg * roll
            ai_team[ai_active]["hp"] = max(0.0, ai_team[ai_active]["hp"] - dmg)
            print(f"👤 {player_team[p_active]['species']} usou {move_name}! Causou {dmg:.1f}% de dano!")
            if ai_team[ai_active]["hp"] <= 0:
                print(f"💥 {ai_team[ai_active]['species']} de Amnesia-AI foi nocauteado!")

        def ai_attack():
            if ai_is_switch or ai_team[ai_active]["hp"] <= 0: return
            move_name = ai_team[ai_active]["moves"][ai_chosen_slot % len(ai_team[ai_active]["moves"])]
            dmg = estimate_damage_pct(ai_team[ai_active]["species"], player_team[p_active]["species"], move_name, {}, {}) * 100.0
            roll = random.uniform(0.85, 1.0)
            dmg = dmg * roll
            player_team[p_active]["hp"] = max(0.0, player_team[p_active]["hp"] - dmg)
            print(f"🤖 {ai_team[ai_active]['species']} de Amnesia-AI usou {move_name}! Causou {dmg:.1f}% de dano!")
            if player_team[p_active]["hp"] <= 0:
                print(f"💥 Seu {player_team[p_active]['species']} foi nocauteado!")

        if p_faster:
            player_attack()
            ai_attack()
        else:
            ai_attack()
            player_attack()


if __name__ == "__main__":
    ckpt = "projects/reinforcement-learning/weights/gen8ou_ppo.pt"
    if len(sys.argv) > 1:
        ckpt = sys.argv[1]
    run_local_battle(ckpt)
