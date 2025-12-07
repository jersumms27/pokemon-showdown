from typing import Any
import json

import numpy as np
from numpy import ndarray

with open("poke-env/pokemon.json") as f:
    POKEMON: dict[str, Any] = json.load(f)
POKEMON_IDX: dict[str, int] = {name.lower(): idx for idx, name in enumerate(POKEMON.keys())}

with open("poke-env/moves.json") as f:
    MOVES: dict[str, Any] = json.load(f)
MOVE_IDX: dict[str, int] = {name.lower(): idx for idx, name in enumerate(MOVES.keys())}

with open("poke-env/items.json") as f:
    ITEMS: dict[str, Any] = json.load(f)
ITEM_IDX: dict[str, int] = {name.lower(): idx for idx, name in enumerate(ITEMS.keys())}

with open("poke-env/abilities.json") as f:
    ABILITIES: dict[str, Any] = json.load(f)
ABILITY_IDX: dict[str, int] = {name.lower(): idx for idx, name in enumerate(ABILITIES.keys())}

CONDITIONS = ['toxic_spikes', 'tailwind', 'spikes', 'light_screen', 'aurora_veil', 'reflect', 'stealth_rock', 'sticky_web']
CONDITION_IDX = {name: idx for idx, name in enumerate(CONDITIONS)}

WEATHERS = ['raindance', 'sandstorm', 'snowscape', 'sunnyday']
WEATHER_IDX = {name: idx for idx, name in enumerate(WEATHERS)}

FIELDS = ['raindance', 'sandstorm', 'snowscape', 'sunnyday']
FIELD_IDX = {name: idx for idx, name in enumerate(FIELDS)}

TYPES = ["bug", "dark", "dragon", "electric", "fairy", "fighting", "fire", "flying", "ghost", "grass", "ground", "ice", "normal", "poison", "psychic", "rock", "steel", "water"]
TYPE_IDX = {name: idx for idx, name in enumerate(TYPES)}

EFFECTS = ['attract', 'battle_bond', 'charge', 'confusion', 'court_change', 'cud_chew', 'dancer', 'destiny_bond', 'disable', 'disguise', 'electric_terrain', 'encore', 'fallen1', 'fallen2', 'fallen3', 'fallen4', 'fallen5', 'fickle_beam', 'flash_fire', 'future_sight', 'glaive_rush', 'hadron_engine', 'heal_bell', 'heal_block', 'ice_face', 'leech_seed', 'leppa_berry', 'magma_storm', 'magnet_rise', 'no_retreat', 'orichalcum_pulse', 'poltergeist', 'protosynthesis', 'protosynthesisatk', 'protosynthesisdef', 'protosynthesisspa', 'psychic_terrain', 'quarkdriveatk', 'quarkdrivedef', 'quarkdrivespe', 'quark_drive', 'salt_cure', 'shed_skin', 'slow_start', 'sticky_hold', 'sticky_web', 'struggle', 'substitute', 'supreme_overlord', 'synchronize', 'taunt', 'tera_shell', 'tera_shift', 'throat_chop', 'tidy_up', 'toxic_debris', 'trapped', 'trick', 'typechange', 'vital_spirit', 'whirlpool', 'yawn', 'zero_to_hero']
EFFECT_IDX = {name: idx for idx, name in enumerate(EFFECTS)}


class State:
    def __init__(self, battle_obj) -> None:
        self.state_dict: dict[str, Any] = self._battle_to_state(battle_obj)
        self.state_json: str = json.dumps(self.state_dict)
        self.state_vector: ndarray = self._dict_to_vector()
    

    def _battle_to_state(self, battle) -> dict[str, Any]:
        obs = battle.current_observation
        state_dict: dict[str, Any] = {}

        state_dict["side_conditions"] = [c.name.lower() for c in obs.side_conditions.keys()]
        state_dict["opp_side_conditions"] = [c.name.lower() for c in obs.opponent_side_conditions.keys()]
        state_dict["weather"] = [w.name.lower() for w in obs.weather.keys()]
        state_dict["weather_turns"] = [t for t in obs.weather.values()]
        state_dict["fields"] = [f.name.lower() for f in obs.fields.keys()]

        state_dict["active_pokemon"] = {}
        state_dict["active_pokemon"]["species"] = obs.active_pokemon.species
        state_dict["active_pokemon"]["type_1"] = POKEMON[obs.active_pokemon.species]["types"][0].lower() if len(POKEMON[obs.active_pokemon.species]["types"]) > 0 else None
        state_dict["active_pokemon"]["type_2"] = POKEMON[obs.active_pokemon.species]["types"][1].lower() if len(POKEMON[obs.active_pokemon.species]["types"]) > 1 else None
        state_dict["active_pokemon"]["level"] = obs.active_pokemon.level
        state_dict["active_pokemon"]["ability"] = obs.active_pokemon.ability
        state_dict["active_pokemon"]["boosts"] = obs.active_pokemon.boosts
        state_dict["active_pokemon"]["hp_fraction"] = obs.active_pokemon.current_hp_fraction
        state_dict["active_pokemon"]["effects"] = [e.name for e in obs.active_pokemon.effects.keys()]
        state_dict["active_pokemon"]["is_dynamaxed"] = obs.active_pokemon.is_dynamaxed
        state_dict["active_pokemon"]["is_terastallized"] = obs.active_pokemon.is_terastallized
        state_dict["active_pokemon"]["item"] = obs.active_pokemon.item
        state_dict["active_pokemon"]["gender"] = obs.active_pokemon.gender.name.lower()
        for i in range(len(obs.active_pokemon.moves)):
            m = list(obs.active_pokemon.moves.values())[i]
            state_dict["active_pokemon"][f"move_{i+1}"] = {}
            state_dict["active_pokemon"][f"move_{i+1}"]["accuracy"] = m.accuracy
            state_dict["active_pokemon"][f"move_{i+1}"]["type"] = m.type.name
            state_dict["active_pokemon"][f"move_{i+1}"]["base_power"] = m.base_power
            state_dict["active_pokemon"][f"move_{i+1}"]["boosts"] = m.boosts
            state_dict["active_pokemon"][f"move_{i+1}"]["category"] = m.category.name
            state_dict["active_pokemon"][f"move_{i+1}"]["crit_ratio"] = m.crit_ratio
        state_dict["active_pokemon"]["tera_type"] = obs.active_pokemon.tera_type.name if obs.active_pokemon.tera_type is not None else None
        state_dict["active_pokemon"]["stats"] = obs.active_pokemon.stats

        state_dict["opp_active_pokemon"] = {}
        state_dict["opp_active_pokemon"]["species"] = obs.opponent_active_pokemon.species
        state_dict["opp_active_pokemon"]["type_1"] = POKEMON[obs.opponent_active_pokemon.species]["types"][0].lower() if len(POKEMON[obs.opponent_active_pokemon.species]["types"]) > 0 else None
        state_dict["opp_active_pokemon"]["type_2"] = POKEMON[obs.opponent_active_pokemon.species]["types"][1].lower() if len(POKEMON[obs.opponent_active_pokemon.species]["types"]) > 1 else None
        state_dict["opp_active_pokemon"]["level"] = obs.opponent_active_pokemon.level
        state_dict["opp_active_pokemon"]["ability"] = obs.opponent_active_pokemon.ability if obs.opponent_active_pokemon.ability is not None else [a.lower() for a in POKEMON[obs.opponent_active_pokemon.species]["abilities"].values()]
        state_dict["opp_active_pokemon"]["boosts"] = obs.opponent_active_pokemon.boosts
        state_dict["opp_active_pokemon"]["hp_fraction"] = obs.opponent_active_pokemon.current_hp_fraction
        state_dict["opp_active_pokemon"]["effects"] = [e.name for e in obs.opponent_active_pokemon.effects.keys()]
        state_dict["opp_active_pokemon"]["is_dynamaxed"] = obs.opponent_active_pokemon.is_dynamaxed
        state_dict["opp_active_pokemon"]["is_terastallized"] = obs.opponent_active_pokemon.is_terastallized
        state_dict["opp_active_pokemon"]["item"] = obs.opponent_active_pokemon.item
        state_dict["opp_active_pokemon"]["gender"] = obs.opponent_active_pokemon.gender.name.lower()
        for i in range(len(obs.opponent_active_pokemon.moves)):
            m = list(obs.opponent_active_pokemon.moves.values())[i]
            state_dict["opp_active_pokemon"][f"move_{i+1}"] = {}
            state_dict["opp_active_pokemon"][f"move_{i+1}"]["accuracy"] = m.accuracy
            state_dict["opp_active_pokemon"][f"move_{i+1}"]["type"] = m.type.name
            state_dict["opp_active_pokemon"][f"move_{i+1}"]["base_power"] = m.base_power
            state_dict["opp_active_pokemon"][f"move_{i+1}"]["boosts"] = m.boosts
            state_dict["opp_active_pokemon"][f"move_{i+1}"]["category"] = m.category.name
            state_dict["opp_active_pokemon"][f"move_{i+1}"]["crit_ratio"] = m.crit_ratio
        state_dict["opp_active_pokemon"]["tera_type"] = obs.opponent_active_pokemon.tera_type.name if obs.opponent_active_pokemon.tera_type is not None else None
        state_dict["opp_active_pokemon"]["stats"] = POKEMON[obs.opponent_active_pokemon.species]["baseStats"]

        for i in range(len(obs.team)):
            p = list(obs.team.values())[i]

            state_dict[f"pokemon_{i+1}"] = {}
            state_dict[f"pokemon_{i+1}"]["species"] = p.species
            state_dict[f"pokemon_{i+1}"]["type_1"] = POKEMON[p.species]["types"][0].lower() if len(POKEMON[p.species]["types"]) > 0 else None
            state_dict[f"pokemon_{i+1}"]["type_2"] = POKEMON[p.species]["types"][1].lower() if len(POKEMON[p.species]["types"]) > 1 else None
            state_dict[f"pokemon_{i+1}"]["level"] = p.level
            state_dict[f"pokemon_{i+1}"]["ability"] = p.ability
            state_dict[f"pokemon_{i+1}"]["boosts"] = p.boosts
            state_dict[f"pokemon_{i+1}"]["hp_fraction"] = p.current_hp_fraction
            state_dict[f"pokemon_{i+1}"]["effects"] = [e.name for e in p.effects.keys()]
            state_dict[f"pokemon_{i+1}"]["is_dynamaxed"] = p.is_dynamaxed
            state_dict[f"pokemon_{i+1}"]["is_terastallized"] = p.is_terastallized
            state_dict[f"pokemon_{i+1}"]["item"] = p.item
            state_dict[f"pokemon_{i+1}"]["gender"] = p.gender.name.lower()
            for j in range(len(p.moves)):
                m = list(p.moves.values())[j]
                state_dict[f"pokemon_{i+1}"][f"move_{j+1}"] = {}
                state_dict[f"pokemon_{i+1}"][f"move_{j+1}"]["name"] = m.id
                state_dict[f"pokemon_{i+1}"][f"move_{j+1}"]["accuracy"] = m.accuracy
                state_dict[f"pokemon_{i+1}"][f"move_{j+1}"]["type"] = m.type.name
                state_dict[f"pokemon_{i+1}"][f"move_{j+1}"]["base_power"] = m.base_power
                state_dict[f"pokemon_{i+1}"][f"move_{j+1}"]["boosts"] = m.boosts
                state_dict[f"pokemon_{i+1}"][f"move_{j+1}"]["category"] = m.category.name
                state_dict[f"pokemon_{i+1}"][f"move_{j+1}"]["crit_ratio"] = m.crit_ratio
            state_dict[f"pokemon_{i+1}"]["tera_type"] = p.tera_type.name if p.tera_type is not None else None
            state_dict[f"pokemon_{i+1}"]["stats"] = p.stats

        for i in range(len(obs.opponent_team)):
            p = list(obs.opponent_team.values())[i]

            state_dict[f"opp_pokemon_{i+1}"] = {}
            state_dict[f"opp_pokemon_{i+1}"]["species"] = p.species
            state_dict[f"opp_pokemon_{i+1}"]["type_1"] = POKEMON[p.species]["types"][0].lower() if len(POKEMON[p.species]["types"]) > 0 else None
            state_dict[f"opp_pokemon_{i+1}"]["type_2"] = POKEMON[p.species]["types"][1].lower() if len(POKEMON[p.species]["types"]) > 1 else None
            state_dict[f"opp_pokemon_{i+1}"]["level"] = p.level
            state_dict[f"opp_pokemon_{i+1}"]["ability"] = p.ability if p.ability is not None else [a.lower() for a in POKEMON[p.species]["abilities"].values()]
            state_dict[f"opp_pokemon_{i+1}"]["boosts"] = p.boosts
            state_dict[f"opp_pokemon_{i+1}"]["hp_fraction"] = p.current_hp_fraction
            state_dict[f"opp_pokemon_{i+1}"]["effects"] = [e.name for e in p.effects.keys()]
            state_dict[f"opp_pokemon_{i+1}"]["is_dynamaxed"] = p.is_dynamaxed
            state_dict[f"opp_pokemon_{i+1}"]["is_terastallized"] = p.is_terastallized
            state_dict[f"opp_pokemon_{i+1}"]["item"] = p.item
            state_dict[f"opp_pokemon_{i+1}"]["gender"] = p.gender.name.lower()
            for j in range(len(p.moves)):
                m = list(p.moves.values())[j]
                state_dict[f"opp_pokemon_{i+1}"][f"move_{j+1}"] = {}
                state_dict[f"opp_pokemon_{i+1}"][f"move_{j+1}"]["name"] = m.id
                state_dict[f"opp_pokemon_{i+1}"][f"move_{j+1}"]["accuracy"] = m.accuracy
                state_dict[f"opp_pokemon_{i+1}"][f"move_{j+1}"]["type"] = m.type.name
                state_dict[f"opp_pokemon_{i+1}"][f"move_{j+1}"]["base_power"] = m.base_power
                state_dict[f"opp_pokemon_{i+1}"][f"move_{j+1}"]["boosts"] = m.boosts
                state_dict[f"opp_pokemon_{i+1}"][f"move_{j+1}"]["category"] = m.category.name
                state_dict[f"opp_pokemon_{i+1}"][f"move_{j+1}"]["crit_ratio"] = m.crit_ratio
            state_dict[f"opp_pokemon_{i+1}"]["tera_type"] = p.tera_type.name if p.tera_type is not None else None
            state_dict[f"opp_pokemon_{i+1}"]["stats"] = POKEMON[p.species]["baseStats"]
        
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

        state_dict["available_moves"] = [m.id for m in battle.available_moves]
        state_dict["available_switches"] = [p.species for p in battle.available_switches]
        state_dict["can_dynamax"] = battle.can_dynamax
        state_dict["can_mega_evolve"] = battle.can_mega_evolve
        state_dict["can_tera"] = battle.can_tera
        state_dict["can_z_move"] = battle.can_z_move
    
        return state_dict
    

    def _dict_to_vector(self) -> ndarray:
        def _get_pokemon_values(pokemon_key: str) -> list[float]:
            pokemon_vector: list[float] = []

            # species
            pokemon_vector.append(POKEMON_IDX[self.state_dict[pokemon_key]["species"]])
            # type 1
            pokemon_vector.append(TYPE_IDX[self.state_dict[pokemon_key]["type_1"]] if self.state_dict[pokemon_key]["type_1"] is not None else -1)
            # type 2
            pokemon_vector.append(TYPE_IDX[self.state_dict[pokemon_key]["type_2"]] if self.state_dict[pokemon_key]["type_2"] is not None else -1)
            # level
            pokemon_vector.append(self.state_dict[pokemon_key]["level"])
            # ability
            pokemon_vector.append(ABILITY_IDX[self.state_dict[pokemon_key]["ability"]] if self.state_dict[pokemon_key]["ability"] in ABILITY_IDX else -1)
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
            # tera type
            pokemon_vector.append(TYPE_IDX[self.state_dict[pokemon_key]["tera_type"]] if self.state_dict[pokemon_key]["tera_type"] is not None else -1)
            # stats:
            # hp
            pokemon_vector.append(self.state_dict[pokemon_key]["stats"]["hp"])
            # attack
            pokemon_vector.append(self.state_dict[pokemon_key]["stats"]["attack"])
            # defense
            pokemon_vector.append(self.state_dict[pokemon_key]["stats"]["defense"])
            # special attack
            pokemon_vector.append(self.state_dict[pokemon_key]["stats"]["special_attack"])
            # special defense
            pokemon_vector.append(self.state_dict[pokemon_key]["stats"]["special_defense"])
            # speed;
            pokemon_vector.append(self.state_dict[pokemon_key]["stats"]["speed"])

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
        # available moves
        vector.extend([MOVE_IDX[m] for m in self.state_dict["available_moves"]] + [-1] * (4 - len(self.state_dict["available_moves"])))
        # available switches
        vector.extend([POKEMON_IDX[s] for s in self.state_dict["available_switches"]] + [-1] * (5 - len(self.state_dict["available_switches"])))
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
    def __init__(self, state: State, action: int, next_state: State, reward: float, terminal: bool = False) -> None:
        self.state: State = state
        self.next_state: State = next_state
        self.action: int = action
        self.reward: float = reward
        self.terminal: bool = terminal
    

    def __str__(self) -> str:
        return str({
            "state": self.state.state_dict,
            "action": self.action,
            "next_state": self.next_state.state_dict,
            "reward": self.reward,
            "terminal": self.terminal
        })