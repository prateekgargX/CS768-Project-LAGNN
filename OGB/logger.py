import torch


class Logger(object):
    def __init__(self, runs, logger, info=None):
        self.ilogger = logger
        logger.info(f"Running with params: {info}") 
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 3
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            self.ilogger.info(f'Run {run + 1:02d}:')
            self.ilogger.info(f'Highest Train: {result[:, 0].max():.2f}')
            self.ilogger.info(f'Highest Valid: {result[:, 1].max():.2f}')
            self.ilogger.info(f'  Final Train: {result[argmax, 0]:.2f}')
            self.ilogger.info(f'   Final Test: {result[argmax, 2]:.2f}')
        else:
            result = 100 * torch.tensor(self.results)

            best_results = []
            for r in result:
                train1 = r[:, 0].max().item()
                valid = r[:, 1].max().item()
                train2 = r[r[:, 1].argmax(), 0].item()
                test = r[r[:, 1].argmax(), 2].item()
                best_results.append((train1, valid, train2, test))

            best_result = torch.tensor(best_results)

            self.ilogger.info(f'All runs:')
            r = best_result[:, 0]
            self.ilogger.info(f'Highest Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 1]
            self.ilogger.info(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 2]
            self.ilogger.info(f'  Final Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 3]
            self.ilogger.info(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')

