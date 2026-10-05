"""CLI-only stubs: no real cases, service, model or network calls."""
import json
import sys

from ragapp import experiments
from ragapp import service as service_module


def test_planner_cli_default_enable_disable_are_independent(tmp_path, monkeypatch, capsys):
    cases = tmp_path / 'synthetic-dev.json'
    cases.write_text(json.dumps([{'id': 'synthetic', 'split': 'dev', 'answerable': False,
                                'question': 'Synthetic question only.'}]), encoding='utf-8')
    settings = tmp_path / 'synthetic-settings.json'
    # The CLI ablation flag explicitly controls planning even if a settings file
    # enables it; omitting the flag must preserve the historical CLI default.
    settings.write_text(json.dumps({'query_planning': True, 'state': 'synthetic-state'}), encoding='utf-8')
    starts = []; observed = []; configurations = []

    class StubService:
        def __init__(self, settings):
            self.settings = settings
            assert settings.load_active is False
            assert settings.state == tmp_path / 'synthetic-state'

        def start(self):
            starts.append(self)
            return self

        def search_detailed(self, *args, **kwargs):
            raise AssertionError('CLI stub must not execute retrieval')

    def stub_run(cases_path, report_path, **options):
        assert cases_path == cases
        assert callable(options['search']) and callable(options['manifest'])
        config = options['configuration']
        observed.append(config['query_planning'])
        assert type(config['query_planning']) is bool
        assert 'stub_mutation' not in config
        configurations.append(config)
        config['stub_mutation'] = str(report_path)
        return {'status': 'synthetic_only', 'metrics': {}}

    monkeypatch.setattr(service_module, 'Service', StubService)
    monkeypatch.setattr(experiments, 'run', stub_run)
    for number, flag in enumerate((None, '--query-planning', '--no-query-planning')):
        argv = ['experiments', '--mode', 'offline', '--settings', str(settings),
                '--cases', str(cases), '--report', str(tmp_path / f'synthetic-report-{number}.json')]
        if flag:
            argv.append(flag)
        monkeypatch.setattr(sys, 'argv', argv)
        experiments.main()
    assert observed == [False, True, False]
    assert len(starts) == 3 and len({id(config) for config in configurations}) == 3
    assert experiments.DEFAULT_CONFIGURATIONS['hybrid_rerank'] == {
        'retrieval_mode': 'hybrid_rerank', 'query_planning': False}
    assert not (tmp_path / 'synthetic-state').exists()
    assert not list(tmp_path.glob('synthetic-report-*.json'))
    assert capsys.readouterr().out.count('synthetic_only') == 3
