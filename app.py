from __future__ import unicode_literals

import multiprocessing
import gunicorn.app.base
import falcon
import apiserver
from gunicorn.six import iteritems


def number_of_workers():
    return (multiprocessing.cpu_count() * 2) + 1


# Gunicorn HTTP server hosting Falcon API framework
class StandaloneApplication(gunicorn.app.base.BaseApplication):

    def __init__(self, api, options=None):
        self.options = options or {}
        self.application = api
        apiserver.config(api)
        super(StandaloneApplication, self).__init__()

    def load_config(self):
        config = dict([(key, value) for key, value in iteritems(self.options)
                       if key in self.cfg.settings and value is not None])
        for key, value in iteritems(config):
            self.cfg.set(key.lower(), value)

    def load(self):
        return self.application


if __name__ == '__main__':
    options = {
        'bind': '%s:%s' % ('0.0.0.0', '8080'),
        'workers': number_of_workers(),
    }
    StandaloneApplication(falcon.API(), options).run()
