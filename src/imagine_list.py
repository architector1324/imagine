import requests

IMAGINE_URL = 'http://{address}/models'


def list_models(args):
    try:
        models = requests.get(IMAGINE_URL.format(address=args.address)).json()['models']

        print('MODELS:')
        for model in models:
            print(model)
    except requests.exceptions.ConnectionError as e:
        print(f'Could not connect to the server. Is it running? Error: {e}')
    except requests.exceptions.RequestException as e:
        print(f'Request failed: {e}')
    except Exception as e:
        print(f'An unexpected error occurred: {e}')
