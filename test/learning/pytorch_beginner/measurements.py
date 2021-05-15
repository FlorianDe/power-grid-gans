import time

from runner import Runner


def run(r: Runner) -> float:
    start = time.time()
    r.run()
    end = time.time()
    return end - start


if __name__ == "__main__":
    runners = {
        '01. WarmUp': __import__('01_warm_up_numpy').WarmUp(),
        '02. PytorchTensors': __import__('02_pytorch_tensors').PytorchTensors(),
        '03. AutogradRunner': __import__('03_autograd').AutogradRunner(),
        '04. CustomAutogradRunner': __import__('04_custom_autograd_fn').CustomAutogradRunner(),
        '05. PytorchNN': __import__('05_pytorch_nn').PytorchNN(),
        '06. PytorchOptim': __import__('06_pytorch_optim').PytorchOptim(),
        '07. PytorchCustomNN': __import__('07_pytorch_custom_nn').PytorchCustomNN(),
        '08. PytorchDynamicNN': __import__('08_pytorch_dynamic_nn').PytorchDynamicNN(),
    }
    for key, runner in runners.items():
        print(f'{key}: {run(runner)}s\n')
