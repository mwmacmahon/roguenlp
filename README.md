Rogue-Like Natural Language Processing (rlnlp) toolkit

MAIN IDEA:
We want to enable a user to interface with a roguelike game using only text commands, such that the player navigates it in a manner akin to a text adventure. Unlike a GPT2-style text adventure, though, this has a consistent core game with deep gameplay underneath it, and the user has access to a visual description of the game at all times!

The structure is hopefully also transferrable across games, though obviously the details of carrying out user commands would need to change significantly.

The proposed method for this project is to break the problem up:
    1. Convert user sentence into one or more function calls (plus arguments)
    2. Using those function calls, execute game commands and/or output direct response messages to the user. Based on what happens, the user's action may need to be cancelled halfway through (a goblin jumps out!), or more information may be requested from user.
    3. [later in project] Describe the happenings onscreen to the user using NLP techniques, until the game UI itself is increasingly unnecessary for play.

The NLP is mostly contained in part 1, but it could also be used in part 3 to give interesting outputs. I have not put as much thought into the latter.

Currently the project is focused around a single model that takes two inputs:
    - User text input (1 sentence, limited # tokens)
        - e.g. "Describe the room"
        - If the user input is in response to a game prompt ("how many?"), we may need to add that context to their input.
    - Map grid representing game environment (N channels, representing a terrain type per channel or an embedding of terrain type, plus the same for player/npcs/objects)
From this, the model should put out:
    - [to start*] A classification of the desired user command
        e.g. "describe_region"
    - A heatmap of the areas of the map that should be targeted by said command,
        e.g. an array equal in size to the input grid, but with ~1 around room and ~0 elsewhere
    - [to start*] Potentially other outputs needed for an algorithm to fully understand the command enough to carry it out. 
        e.g. an arbitrary length list of specific targets in the tile locations.

*Note that the model output might need to fundamentally change based on the practical necessities of carrying out various user commands. I strongly suspect that instead of outputting one single command via classification, plus additional info via a second method (e.g. dense layer), it makes more sense to replace both of those with a sequence of commands and command-arguments via LSTM/RNN/etc. However, this is perhaps a bit complex for early prototypes!

The reason we use a heatmap output instead of simply outputting coordinates is that it it seems to be difficult/inefficient to regress out coordinates, and doubly so for identifying regions. There are techniques in image processing for this, but it's not clear to me they are good fits for our situation. My hope is that it shouldn't be too hard for each specific algorithm to parse the heatmap output, since said output is trained to produce what the algorith expects. 
Example 1: If the command is "move_to_wall", the algorithm can pick the wall-adjacent movement tile in the heatmap's high-value region that is closest to the player.
Example 2: If the command is "describe_region", the algorithm runs the game's "look" command on each terrain type in the heatmap's high-value region (including walls), plus each item in the region, and tries to output all of that information in a human-parsable way (this is where using NLP output would actually come in quite handy in a final product).

CURRENT OBJECTIVES: 
- Create a training dataset generator. Must be able to produce input grids and user text, plus output heatmaps and interpreted commands, en masse, in a way that can easily be added onto. This will probably need to be charted out by hand first, as it could be a doozy. For example, it might need systems like:
    - Subsystems for different commands to train (plus non-command unusable text)
    - A library of functions to generate inputs from growing lists of substrings, probably factored out to a ridiculous degree so that each can pick from a random sampling of its potential more-specific implmentations (e.g. location_query_creator(), location_query_text_creator(), location_query_text_creator_syntax_1())
    - A system to pass information around between these functions (no matter the command, we need something to take grids -> heatmaps for the right targets).
- Create scripts to define model, create datasets, run training, run hyperparameter optimization, etc., using best-practices formats. I plan on using tf.keras but I'm not married to it...thankfully huggingface is cross-platform and things like nltk can be used for preprocessing regardless of later stages of pipeline.

NOTE ON CURRENT PROJECT STATE::
In this early stage, we just have a scripts folder containing a bag of scripts for testing protoype models. Future versions will be more structured and installable (src/rlnlp directory w/ __init__.py files, setup.py, etc.). I'd like a good infrastructure that's compatible with cloud computing services.

Similarly, the Git structure is currently just a master branch, develop branch, and working branches - a more nuanced system involving feature branches, etc., should be implemented once the project has taken shape.



