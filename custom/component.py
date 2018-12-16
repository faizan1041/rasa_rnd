from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

import typing
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Text

from rasa_nlu.components import Component
from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.training_data import Message
from rasa_nlu.training_data import TrainingData

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from spacy.language import Language
    from rasa_nlu.model import Metadata

from rasa_nlu.utils.spacy_utils import SpacyNLP
class SpacyNLPCustom(SpacyNLP):
    name = "nlp_spacy_custom"
    def __init__(self, component_config=None, nlp=None):
        # type: (Dict[Text, Any], Language) -> None
    
        self.nlp = nlp
        super(SpacyNLPCustom, self).__init__(component_config)

    @classmethod
    def create(cls, cfg):
        # type: (RasaNLUModelConfig) -> SpacyNLP
        import spacy

        component_conf = cfg.for_component(cls.name, cls.defaults)
        spacy_model_name = component_conf.get("model")
        # if no model is specified, we fall back to the language string
        if not spacy_model_name:
            spacy_model_name = cfg.language
            component_conf["model"] = cfg.language
        logger.info("Trying to load spacy model with "
                    "name '{}'".format(spacy_model_name))

        # import pdb; pdb.set_trace()

        nlp = spacy.load(spacy_model_name, parser=False)
        cls.ensure_proper_language_model(nlp)
        return SpacyNLP(component_conf, nlp)