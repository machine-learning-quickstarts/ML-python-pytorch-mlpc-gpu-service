# Import your handlers here
from service import MNIST, Intro


# Configuration for web API implementation
def config(api):
    # Instantiate handlers
    intro = Intro()
    mnist = MNIST()

    # Map routes
    api.add_route('/mnist', intro)
    api.add_route('/mnist/{index:int(min=0)}', mnist)
