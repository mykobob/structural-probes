import json, requests

class StanfordCoreNLP:

    """
    Modified from https://github.com/smilli/py-corenlp (https://github.com/smilli/py-corenlp)
    """
    def __init__(self, server_url):
        # TODO: Error handling? More checking on the url?
        if server_url[-1] == '/':
            server_url = server_url[:-1]
        self.server_url = server_url

    def annotate(self, text, properties=None):
        assert isinstance(text, str)

        if properties is None:
            properties = {}
        else:
            assert isinstance(properties, dict)

        # Checks that the Stanford CoreNLP server is started.
        try:
            requests.get(self.server_url)
        except requests.exceptions.ConnectionError:
            raise Exception('Check whether you have started the CoreNLP server e.g.\n'
                                '$ cd <path_to_core_nlp_folder>/stanford-corenlp-full-2016-10-31/ \n'
                '$ java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port <port>' )

        data = text.encode()
        r = requests.post(
            self.server_url, params={
                    'properties': str(properties)
            }, data=data, headers={'Connection': 'close'})

        output = r.text

        if ('outputFormat' in properties
            and properties['outputFormat'] == 'json'):
            try:
                output = json.loads(output, encoding='utf-8', strict=True)
            except:
                pass
        return output