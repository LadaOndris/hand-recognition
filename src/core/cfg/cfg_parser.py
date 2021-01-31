import os
import re


class CfgParser():

    def __init__(self):
        self._supported_layer_types = ''
        return

    def parse(self, cfg_file_path, custom_layers={}):
        """
        Parses cfg file and returns a model.

        Parameters
        ----------
        cfg_file_path : String
            File containing network architecture in a cfg format.
        custom_layers : TYPE, optional
            Custom layers in the cfg file. The key is a name of the layer in cfg file
            and the value is a subclass of tf.keras.layers.Layer. The default is {}.


        """
        if not os.path.isfile(cfg_file_path) or not os.access(cfg_file_path, os.R_OK):
            raise OSError(F"File '{cfg_file_path} doesn't exist or is not accessible.'")
        self._custom_layers = custom_layers

        self._read_file_content(cfg_file_path)
        self._preprocess_file_content()
        self._parse_content_into_blocks()
        self._create_layers_from_blocks()
        return self._create_model()

    def _read_file_content(self, cfg_file_path):
        """
        Reads file content and saves all lines into an array.
        """
        with open(cfg_file_path, 'r') as file:
            self._file_content = file.readlines()

    def _preprocess_file_content(self):
        """
        Preprocesses the content in such a way that 
        each line contains a meaningful information and without 
        comments and superfluous whitespaces.
        """

        # Replace all whitespaces from each line.
        for i in range(len(self._file_content)):
            # Replace all whitespaces.
            self._file_content[i] = re.sub(r"\s+", "", self._file_content[i], flags=re.UNICODE)
            # Remove comments.
            self._file_content[i] = self._file_content[i].split('#')[0]

        # Remove all empty lines.
        self._file_content = [line for line in self._file_content if len(line) > 0]

    def _parse_content_into_blocks(self):
        block = {}
        self._blocks = []

        for i, line in enumerate(self._file_content):

            if line[0] == '[':  # indicates new block
                if line[-1] != ']':
                    raise ValueError("Incorrectly enclosed block type." +
                                     "Line cannot start with '[' and not end with ']'.")
                if len(line[1:-1]) == 0:
                    raise ValueError("Missing block type. Found empty brackets '[]'.")

                # Don't append an empty block.
                # But allow empty blocks in cfg file without any properties. 
                # Default values can be used in such cases.
                if i != 0:
                    self._blocks.append(block)
                    block.clear()
                block["type"] = line[1:-1]
            else:  # atribute of a block
                key, value = line.split("=")
                block[key] = value

        self._blocks.append(block)

    def _create_layers_from_blocks(self):
        for block in self._blocks:
            block_type = block['type']
            # if not self._block_type_exists(block['type']):
            #    raise ValueError("Unknown block type {block['type']}.")

            if block_type == 'convolutional':
                self._create_layer_convolutional(block)
            # elif block_type == 'upsample':
            # elif block_type == 'maxpool':
            # elif block_type == 'route':
            # elif block_type == 'shortcut':
            elif block_type == '':
                pass

    def _create_layer_convolutional(self, block):
        pass

    def _create_model(self):

        return
