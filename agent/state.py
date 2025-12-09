from typing import Any
import json

import numpy as np
from numpy import ndarray

def _normalize(s: str) -> str:
    return s.lower().replace(" ", "")

with open("poke-env/pokemon.json") as f:
    POKEMON: dict[str, Any] = json.load(f)
POKEMON_IDX: dict[str, int] = {_normalize(name): idx for idx, name in enumerate([p["name"] for p in POKEMON.values()])}

def _get_pokemon(name: str) -> dict[str, Any]:
    n: str = _normalize(name)
    if n in POKEMON:
        return POKEMON[n]
    
    for k in POKEMON.keys():
        if k in n:
            return POKEMON[k]
    
    return {}

def _get_pokemon_idx(p: str) -> int:
    if p in POKEMON_IDX:
        return POKEMON_IDX[p]
    
    for k in POKEMON_IDX.keys():
        if k in p:
            return POKEMON_IDX[k]
    
    return -1

with open("poke-env/moves.json") as f:
    MOVES: dict[str, Any] = json.load(f)
MOVE_IDX: dict[str, int] = {name: idx for idx, name in enumerate(list(MOVES.keys()))}

with open("poke-env/items.json") as f:
    ITEMS: dict[str, Any] = json.load(f)
ITEM_IDX: dict[str, int] = {name: idx for idx, name in enumerate([i["name"] for i in ITEMS.values()])}

with open("poke-env/abilities.json") as f:
    ABILITIES: dict[str, Any] = json.load(f)
ABILITY_IDX: dict[str, int] = {name: idx for idx, name in enumerate(ABILITIES.keys())}

CONDITIONS = ['toxic_spikes', 'tailwind', 'spikes', 'light_screen', 'aurora_veil', 'reflect', 'stealth_rock', 'sticky_web']
CONDITION_IDX = {name: idx for idx, name in enumerate(CONDITIONS)}

WEATHERS = ['raindance', 'sandstorm', 'snowscape', 'sunnyday']
WEATHER_IDX = {name: idx for idx, name in enumerate(WEATHERS)}

FIELDS = ['electric_terrain', 'grassy_terrain', 'psychic_terrain', 'trick_room']
FIELD_IDX = {name: idx for idx, name in enumerate(FIELDS)}

TYPES = ["bug", "dark", "dragon", "electric", "fairy", "fighting", "fire", "flying", "ghost", "grass", "ground", "ice", "normal", "poison", "psychic", "rock", "steel", "water", "three_question_marks", "stellar"]
TYPE_IDX = {name: idx for idx, name in enumerate(TYPES)}

EFFECTS = ['attract', 'battle_bond', 'charge', 'confusion', 'court_change', 'cud_chew', 'dancer', 'destiny_bond', 'disable', 'disguise', 'electric_terrain', 'encore', 'fallen1', 'fallen2', 'fallen3', 'fallen4', 'fallen5', 'fickle_beam', 'flash_fire', 'future_sight', 'glaive_rush', 'hadron_engine', 'heal_bell', 'heal_block', 'ice_face', 'leech_seed', 'leppa_berry', 'magma_storm', 'magnet_rise', 'no_retreat', 'orichalcum_pulse', 'poltergeist', 'protosynthesis', 'protosynthesisatk', 'protosynthesisdef', 'protosynthesisspa', 'psychic_terrain', 'quarkdriveatk', 'quarkdrivedef', 'quarkdrivespe', 'quark_drive', 'salt_cure', 'shed_skin', 'slow_start', 'sticky_hold', 'sticky_web', 'struggle', 'substitute', 'supreme_overlord', 'synchronize', 'taunt', 'tera_shell', 'tera_shift', 'throat_chop', 'tidy_up', 'toxic_debris', 'trapped', 'trick', 'typechange', 'vital_spirit', 'whirlpool', 'yawn', 'zero_to_hero']
EFFECT_IDX = {name: idx for idx, name in enumerate(EFFECTS)}

MOVE_CATS = ["physical", "special", "status"]
MOVE_CAT_IDX = {name: idx for idx, name in enumerate(MOVE_CATS)}


class State:
    def __init__(self, battle_obj) -> None:
        self.state_dict: dict[str, Any] = self._battle_to_state(battle_obj)
        self.state_json: str = json.dumps(self.state_dict)
        self.state_vector: ndarray = self._dict_to_vector()
    

    def _battle_to_state(self, battle) -> dict[str, Any]:
        obs = battle.current_observation
        state_dict: dict[str, Any] = {}

        state_dict["side_conditions"] = [c.name for c in obs.side_conditions.keys()]
        state_dict["opp_side_conditions"] = [c.name for c in obs.opponent_side_conditions.keys()]
        state_dict["weather"] = [_normalize(w.name) for w in obs.weather.keys()]
        state_dict["weather_turns"] = [t for t in obs.weather.values()]
        state_dict["fields"] = [f.name for f in obs.fields.keys()]

        state_dict["active_pokemon"] = {}
        state_dict["active_pokemon"]["species"] = obs.active_pokemon.species
        state_dict["active_pokemon"]["type_1"] = _normalize(_get_pokemon(obs.active_pokemon.species)["types"][0]) if len(_get_pokemon(obs.active_pokemon.species)["types"]) > 0 else None
        state_dict["active_pokemon"]["type_2"] = _normalize(_get_pokemon(obs.active_pokemon.species)["types"][1]) if len(_get_pokemon(obs.active_pokemon.species)["types"]) > 1 else None
        state_dict["active_pokemon"]["level"] = obs.active_pokemon.level

        state_dict["active_pokemon"]["ability"] = obs.active_pokemon.ability if obs.active_pokemon.ability is not None else [_normalize(a) for a in _get_pokemon(obs.active_pokemon.species)["abilities"].values()]
        state_dict["active_pokemon"]["boosts"] = obs.active_pokemon.boosts
        state_dict["active_pokemon"]["hp_fraction"] = obs.active_pokemon.current_hp_fraction
        state_dict["active_pokemon"]["effects"] = [e.name for e in obs.active_pokemon.effects.keys()]
        state_dict["active_pokemon"]["is_dynamaxed"] = obs.active_pokemon.is_dynamaxed
        state_dict["active_pokemon"]["is_terastallized"] = obs.active_pokemon.is_terastallized
        state_dict["active_pokemon"]["item"] = obs.active_pokemon.item
        state_dict["active_pokemon"]["gender"] = obs.active_pokemon.gender.name
        for i in range(len(obs.active_pokemon.moves)):
            m = list(obs.active_pokemon.moves.values())[i]
            state_dict["active_pokemon"][f"move_{i+1}"] = {}
            state_dict["active_pokemon"][f"move_{i+1}"]["name"] = m.id
            state_dict["active_pokemon"][f"move_{i+1}"]["accuracy"] = m.accuracy
            state_dict["active_pokemon"][f"move_{i+1}"]["type"] = _normalize(m.type.name)
            state_dict["active_pokemon"][f"move_{i+1}"]["base_power"] = m.base_power
            state_dict["active_pokemon"][f"move_{i+1}"]["category"] = _normalize(m.category.name)
            state_dict["active_pokemon"][f"move_{i+1}"]["crit_ratio"] = m.crit_ratio
        state_dict["active_pokemon"]["tera_type"] = _normalize(obs.active_pokemon.tera_type.name) if obs.active_pokemon.tera_type is not None else None
        state_dict["active_pokemon"]["stats"] = obs.active_pokemon.stats

        state_dict["opp_active_pokemon"] = {}
        state_dict["opp_active_pokemon"]["species"] = obs.opponent_active_pokemon.species
        state_dict["opp_active_pokemon"]["type_1"] = _normalize(_get_pokemon(obs.opponent_active_pokemon.species)["types"][0]) if len(_get_pokemon(obs.opponent_active_pokemon.species)["types"]) > 0 else None
        state_dict["opp_active_pokemon"]["type_2"] = _normalize(_get_pokemon(obs.opponent_active_pokemon.species)["types"][1]) if len(_get_pokemon(obs.opponent_active_pokemon.species)["types"]) > 1 else None
        state_dict["opp_active_pokemon"]["level"] = obs.opponent_active_pokemon.level
        state_dict["opp_active_pokemon"]["ability"] = obs.opponent_active_pokemon.ability if obs.opponent_active_pokemon.ability is not None else [_normalize(a) for a in _get_pokemon(obs.opponent_active_pokemon.species)["abilities"].values()]
        state_dict["opp_active_pokemon"]["boosts"] = obs.opponent_active_pokemon.boosts
        state_dict["opp_active_pokemon"]["hp_fraction"] = obs.opponent_active_pokemon.current_hp_fraction
        state_dict["opp_active_pokemon"]["effects"] = [e.name for e in obs.opponent_active_pokemon.effects.keys()]
        state_dict["opp_active_pokemon"]["is_dynamaxed"] = obs.opponent_active_pokemon.is_dynamaxed
        state_dict["opp_active_pokemon"]["is_terastallized"] = obs.opponent_active_pokemon.is_terastallized
        state_dict["opp_active_pokemon"]["item"] = obs.opponent_active_pokemon.item
        state_dict["opp_active_pokemon"]["gender"] = obs.opponent_active_pokemon.gender.name
        for i in range(len(obs.opponent_active_pokemon.moves)):
            m = list(obs.opponent_active_pokemon.moves.values())[i]
            state_dict["opp_active_pokemon"][f"move_{i+1}"] = {}
            state_dict["opp_active_pokemon"][f"move_{i+1}"]["name"] = m.id
            state_dict["opp_active_pokemon"][f"move_{i+1}"]["accuracy"] = m.accuracy
            state_dict["opp_active_pokemon"][f"move_{i+1}"]["type"] = _normalize(m.type.name)
            state_dict["opp_active_pokemon"][f"move_{i+1}"]["base_power"] = m.base_power
            state_dict["opp_active_pokemon"][f"move_{i+1}"]["category"] = _normalize(m.category.name)
            state_dict["opp_active_pokemon"][f"move_{i+1}"]["crit_ratio"] = m.crit_ratio
        state_dict["opp_active_pokemon"]["tera_type"] = _normalize(obs.opponent_active_pokemon.tera_type.name) if obs.opponent_active_pokemon.tera_type is not None else None
        state_dict["opp_active_pokemon"]["stats"] = _get_pokemon(obs.opponent_active_pokemon.species)["baseStats"]

        for i in range(len(obs.team)):
            p = list(obs.team.values())[i]

            state_dict[f"pokemon_{i+1}"] = {}
            state_dict[f"pokemon_{i+1}"]["species"] = p.species
            state_dict[f"pokemon_{i+1}"]["type_1"] = _normalize(_get_pokemon(p.species)["types"][0]) if len(_get_pokemon(p.species)["types"]) > 0 else None
            state_dict[f"pokemon_{i+1}"]["type_2"] = _normalize(_get_pokemon(p.species)["types"][1]) if len(_get_pokemon(p.species)["types"]) > 1 else None
            state_dict[f"pokemon_{i+1}"]["level"] = p.level
            state_dict[f"pokemon_{i+1}"]["ability"] = p.ability if p.ability is not None else [_normalize(a) for a in _get_pokemon(p.species)["abilities"].values()]
            state_dict[f"pokemon_{i+1}"]["boosts"] = p.boosts
            state_dict[f"pokemon_{i+1}"]["hp_fraction"] = p.current_hp_fraction
            state_dict[f"pokemon_{i+1}"]["effects"] = [e.name for e in p.effects.keys()]
            state_dict[f"pokemon_{i+1}"]["is_dynamaxed"] = p.is_dynamaxed
            state_dict[f"pokemon_{i+1}"]["is_terastallized"] = p.is_terastallized
            state_dict[f"pokemon_{i+1}"]["item"] = p.item
            state_dict[f"pokemon_{i+1}"]["gender"] = p.gender.name
            for j in range(len(p.moves)):
                m = list(p.moves.values())[j]
                state_dict[f"pokemon_{i+1}"][f"move_{j+1}"] = {}
                state_dict[f"pokemon_{i+1}"][f"move_{j+1}"]["name"] = m.id
                state_dict[f"pokemon_{i+1}"][f"move_{j+1}"]["accuracy"] = m.accuracy
                state_dict[f"pokemon_{i+1}"][f"move_{j+1}"]["type"] = _normalize(m.type.name)
                state_dict[f"pokemon_{i+1}"][f"move_{j+1}"]["base_power"] = m.base_power
                state_dict[f"pokemon_{i+1}"][f"move_{j+1}"]["category"] = _normalize(m.category.name)
                state_dict[f"pokemon_{i+1}"][f"move_{j+1}"]["crit_ratio"] = m.crit_ratio
            state_dict[f"pokemon_{i+1}"]["tera_type"] = _normalize(p.tera_type.name) if p.tera_type is not None else None
            state_dict[f"pokemon_{i+1}"]["stats"] = p.stats

        for i in range(len(obs.opponent_team)):
            p = list(obs.opponent_team.values())[i]

            state_dict[f"opp_pokemon_{i+1}"] = {}
            state_dict[f"opp_pokemon_{i+1}"]["species"] = p.species
            state_dict[f"opp_pokemon_{i+1}"]["type_1"] = _normalize(_get_pokemon(p.species)["types"][0]) if len(_get_pokemon(p.species)["types"]) > 0 else None
            state_dict[f"opp_pokemon_{i+1}"]["type_2"] = _normalize(_get_pokemon(p.species)["types"][1]) if len(_get_pokemon(p.species)["types"]) > 1 else None
            state_dict[f"opp_pokemon_{i+1}"]["level"] = p.level
            state_dict[f"opp_pokemon_{i+1}"]["ability"] = p.ability if p.ability is not None else [_normalize(a) for a in _get_pokemon(p.species)["abilities"].values()]
            state_dict[f"opp_pokemon_{i+1}"]["boosts"] = p.boosts
            state_dict[f"opp_pokemon_{i+1}"]["hp_fraction"] = p.current_hp_fraction
            state_dict[f"opp_pokemon_{i+1}"]["effects"] = [e.name for e in p.effects.keys()]
            state_dict[f"opp_pokemon_{i+1}"]["is_dynamaxed"] = p.is_dynamaxed
            state_dict[f"opp_pokemon_{i+1}"]["is_terastallized"] = p.is_terastallized
            state_dict[f"opp_pokemon_{i+1}"]["item"] = p.item
            state_dict[f"opp_pokemon_{i+1}"]["gender"] = p.gender.name
            for j in range(len(p.moves)):
                m = list(p.moves.values())[j]
                state_dict[f"opp_pokemon_{i+1}"][f"move_{j+1}"] = {}
                state_dict[f"opp_pokemon_{i+1}"][f"move_{j+1}"]["name"] = m.id
                state_dict[f"opp_pokemon_{i+1}"][f"move_{j+1}"]["accuracy"] = m.accuracy
                state_dict[f"opp_pokemon_{i+1}"][f"move_{j+1}"]["type"] = _normalize(m.type.name)
                state_dict[f"opp_pokemon_{i+1}"][f"move_{j+1}"]["base_power"] = m.base_power
                state_dict[f"opp_pokemon_{i+1}"][f"move_{j+1}"]["category"] = _normalize(m.category.name)
                state_dict[f"opp_pokemon_{i+1}"][f"move_{j+1}"]["crit_ratio"] = m.crit_ratio
            state_dict[f"opp_pokemon_{i+1}"]["tera_type"] = _normalize(p.tera_type.name) if p.tera_type is not None else None
            state_dict[f"opp_pokemon_{i+1}"]["stats"] = _get_pokemon(p.species)["baseStats"]
        
        state_dict["turn"] = battle.turn
        state_dict["force_switch"] = battle.force_switch
        state_dict["trapped"] = battle.trapped
        state_dict["maybe_trapped"] = battle.maybe_trapped
        state_dict["used_dynamax"] = battle.used_dynamax
        state_dict["used_mega_evolve"] = battle.used_mega_evolve
        state_dict["used_tera"] = battle.used_tera
        state_dict["used_z_move"] = battle.used_z_move
        state_dict["dynamax_turns_left"] = battle.dynamax_turns_left
        state_dict["opponent_dynamax_turns_left"] = battle.opponent_dynamax_turns_left

        state_dict["battle_finished"] = battle.finished
        state_dict["won"] = battle.won

        state_dict["available_moves"] = [m.id for m in battle.available_moves]
        state_dict["available_switches"] = [p.species for p in battle.available_switches]
        state_dict["can_dynamax"] = battle.can_dynamax
        state_dict["can_mega_evolve"] = battle.can_mega_evolve
        state_dict["can_tera"] = battle.can_tera
        state_dict["can_z_move"] = battle.can_z_move
    
        return state_dict
    

    def _dict_to_vector(self) -> ndarray:
        def _get_move_values(pokemon_key: str, move_key: str) -> list[float]:
            if move_key not in self.state_dict[pokemon_key]:
                return [-1] * 8

            move_vector: list[float] = []

            # name
            move_vector.append(MOVE_IDX[self.state_dict[pokemon_key][move_key]["name"]])
            # accuracy
            move_vector.append(self.state_dict[pokemon_key][move_key]["accuracy"])
            # type
            move_vector.append(TYPE_IDX[self.state_dict[pokemon_key][move_key]["type"]])
            # base power
            move_vector.append(self.state_dict[pokemon_key][move_key]["base_power"])
            # category
            move_vector.extend([float(self.state_dict[pokemon_key][move_key]["category"] == MOVE_CATS[i]) for i in range(len(MOVE_CATS))])
            # crit ratio
            move_vector.append(self.state_dict[pokemon_key][move_key]["crit_ratio"])

            return move_vector

        def _get_pokemon_values(pokemon_key: str) -> list[float]:
            if pokemon_key not in self.state_dict:
                return [-1] * 124

            pokemon_vector: list[float] = []

            # species
            pokemon_vector.append(_get_pokemon_idx(self.state_dict[pokemon_key]["species"]))
            # type 1
            pokemon_vector.append(TYPE_IDX[self.state_dict[pokemon_key]["type_1"]] if self.state_dict[pokemon_key]["type_1"] is not None else -1)
            # type 2
            pokemon_vector.append(TYPE_IDX[self.state_dict[pokemon_key]["type_2"]] if self.state_dict[pokemon_key]["type_2"] is not None else -1)
            # level
            pokemon_vector.append(self.state_dict[pokemon_key]["level"])
            # ability
            if isinstance(self.state_dict[pokemon_key]["ability"], str):
                pokemon_vector.extend([ABILITY_IDX[self.state_dict[pokemon_key]["ability"]]] + [-1] * 5)
            else: # isinstance(self.state_dict[pokemon_key]["ability"], list)
                pokemon_vector.extend([ABILITY_IDX[self.state_dict[pokemon_key]["ability"][i]] for i in range(len(self.state_dict[pokemon_key]["ability"]))] + [-1] * (6 - len(self.state_dict[pokemon_key]["ability"])))
            # boosts:
            # accuracy
            pokemon_vector.append(self.state_dict[pokemon_key]["boosts"]["accuracy"])
            # attack
            pokemon_vector.append(self.state_dict[pokemon_key]["boosts"]["atk"])
            # defense
            pokemon_vector.append(self.state_dict[pokemon_key]["boosts"]["def"])
            # evasion
            pokemon_vector.append(self.state_dict[pokemon_key]["boosts"]["evasion"])
            # special attack
            pokemon_vector.append(self.state_dict[pokemon_key]["boosts"]["spa"])
            # special defense
            pokemon_vector.append(self.state_dict[pokemon_key]["boosts"]["spd"])
            # speed;
            pokemon_vector.append(self.state_dict[pokemon_key]["boosts"]["spe"])
            # hp fraction
            pokemon_vector.append(self.state_dict[pokemon_key]["hp_fraction"])
            # effects
            pokemon_vector.extend([float(e in self.state_dict[pokemon_key]["effects"]) for e in EFFECTS])
            # is dynamaxed
            pokemon_vector.append(float(self.state_dict[pokemon_key]["is_dynamaxed"] == True))
            # is terestallized
            pokemon_vector.append(float(self.state_dict[pokemon_key]["is_terastallized"] == True))
            # item
            pokemon_vector.append(ITEM_IDX[self.state_dict[pokemon_key]["item"]] if self.state_dict[pokemon_key]["item"] in ITEM_IDX else -1)
            # gender
            pokemon_vector.append(float(self.state_dict[pokemon_key]["gender"] == "male"))
            # moves
            for i in range(4):
                pokemon_vector.extend(_get_move_values(pokemon_key, f"move_{i+1}"))
            # tera type
            pokemon_vector.append(TYPE_IDX[self.state_dict[pokemon_key]["tera_type"]] if self.state_dict[pokemon_key]["tera_type"] is not None else -1)
            # stats:
            # hp
            pokemon_vector.append(self.state_dict[pokemon_key]["stats"]["hp"])
            # attack
            pokemon_vector.append(self.state_dict[pokemon_key]["stats"]["atk"])
            # defense
            pokemon_vector.append(self.state_dict[pokemon_key]["stats"]["def"])
            # special attack
            pokemon_vector.append(self.state_dict[pokemon_key]["stats"]["spa"])
            # special defense
            pokemon_vector.append(self.state_dict[pokemon_key]["stats"]["spd"])
            # speed;
            pokemon_vector.append(self.state_dict[pokemon_key]["stats"]["spe"])

            return pokemon_vector

        vector: list[float] = []
        # side conditions
        vector.extend([float(c in self.state_dict["side_conditions"]) for c in CONDITIONS])
        # opponent side conditions
        vector.extend([float(c in self.state_dict["opp_side_conditions"]) for c in CONDITIONS])
        # weather
        vector.extend([float(w in self.state_dict["weather"]) for w in WEATHERS])
        # weather turns
        vector.extend([0.0] * len(WEATHERS))
        if len(self.state_dict["weather"]) > 0:
            vector[-(WEATHER_IDX[self.state_dict["weather"][0]] + 1)] = self.state_dict["weather_turns"][0]
        # fields
        vector.extend([float(f in self.state_dict["fields"]) for f in FIELDS])
        # active pokemon
        vector.extend(_get_pokemon_values("active_pokemon"))
        # opponent active pokemon
        vector.extend(_get_pokemon_values("opp_active_pokemon"))
        # pokemon 1
        vector.extend(_get_pokemon_values("pokemon_1"))
        # pokemon 2
        vector.extend(_get_pokemon_values("pokemon_2"))
        # pokemon 3
        vector.extend(_get_pokemon_values("pokemon_3"))
        # pokemon 4
        vector.extend(_get_pokemon_values("pokemon_4"))
        # pokemon 5
        vector.extend(_get_pokemon_values("pokemon_5"))
        # pokemon 6
        vector.extend(_get_pokemon_values("pokemon_6"))
        # opponent pokemon 1
        vector.extend(_get_pokemon_values("opp_pokemon_1"))
        # opponent pokemon 2
        vector.extend(_get_pokemon_values("opp_pokemon_2"))
        # opponent pokemon 3
        vector.extend(_get_pokemon_values("opp_pokemon_3"))
        # opponent pokemon 4
        vector.extend(_get_pokemon_values("opp_pokemon_4"))
        # opponent pokemon 5
        vector.extend(_get_pokemon_values("opp_pokemon_5"))
        # opponent pokemon 6
        vector.extend(_get_pokemon_values("opp_pokemon_6"))
        # turn
        vector.append(self.state_dict["turn"])
        # force switch
        vector.append(float(self.state_dict["force_switch"] == True))
        # trapped
        vector.append(float(self.state_dict["trapped"] == True))
        # maybe trapped
        vector.append(float(self.state_dict["maybe_trapped"] == True))
        # used dynamax
        vector.append(float(self.state_dict["used_dynamax"] == True))
        # used mega evolution
        vector.append(float(self.state_dict["used_mega_evolve"] == True))
        # used tera
        vector.append(float(self.state_dict["used_tera"] == True))
        # used z move
        vector.append(float(self.state_dict["used_z_move"] == True))
        # dynamax turns left
        vector.append(self.state_dict["dynamax_turns_left"])
        # opponent dynamax turns left
        vector.append(self.state_dict["opponent_dynamax_turns_left"])
        # battle finished
        vector.append(float(self.state_dict["battle_finished"] == True))
        # battle won
        vector.append(float(self.state_dict["won"] == True))
        # available moves
        vector.extend([MOVE_IDX[m] for m in self.state_dict["available_moves"]] + [-1] * (4 - len(self.state_dict["available_moves"])))
        # available switches
        vector.extend([_get_pokemon_idx(s) for s in self.state_dict["available_switches"]] + [-1] * (5 - len(self.state_dict["available_switches"])))
        # can dynamax
        vector.append(float(self.state_dict["can_dynamax"] == True))
        # can mega evolve
        vector.append(float(self.state_dict["can_mega_evolve"] == True))
        # can tera
        vector.append(float(self.state_dict["can_tera"] == True))
        # can z move
        vector.append(float(self.state_dict["can_z_move"] == True))

        return np.array(vector, dtype=np.float32)
    

    def __str__(self) -> str:
        return self.state_json


class Transition:
    def __init__(self, state: State, action: int, next_state: State, terminal: bool = False) -> None:
        self.state: State = state
        self.next_state: State = next_state
        self.action: int = action
        self.reward: float = self._calculate_reward()
        self.terminal: bool = terminal
    

    def _calculate_reward(self) -> float:
        num_pokemon_1: int = sum([1 if f"pokemon_{i+1}" in self.state.state_dict else 0 for i in range(6)])
        num_pokemon_2: int = sum([1 if f"pokemon_{i+1}" in self.next_state.state_dict else 0 for i in range(6)])
        num_opp_pokemon_1: int = sum([1 if f"opp_pokemon_{i+1}" in self.state.state_dict else 0 for i in range(6)])
        num_opp_pokemon_2: int = sum([1 if f"opp_pokemon_{i+1}" in self.next_state.state_dict else 0 for i in range(6)])

        health_1: float = sum([self.state.state_dict[f"pokemon_{i+1}"]["hp_fraction"] for i in range(num_pokemon_1)]) / float(num_pokemon_1)
        health_2: float = sum([self.next_state.state_dict[f"pokemon_{i+1}"]["hp_fraction"] for i in range(num_pokemon_2)]) / float(num_pokemon_2)
        opp_health_1: float = sum([self.state.state_dict[f"opp_pokemon_{i+1}"]["hp_fraction"] for i in range(num_opp_pokemon_1)]) / float(num_opp_pokemon_1)
        opp_health_2: float = sum([self.next_state.state_dict[f"opp_pokemon_{i+1}"]["hp_fraction"] for i in range(num_opp_pokemon_2)]) / float(num_opp_pokemon_2)

        return (health_2 - health_1) - (opp_health_2 - opp_health_1)
    

    def __str__(self) -> str:
        return str({
            "state": self.state.state_dict,
            "action": self.action,
            "next_state": self.next_state.state_dict,
            "reward": self.reward,
            "terminal": self.terminal
        })