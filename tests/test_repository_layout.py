"""Regression checks for the relocated workflows and shared evaluator."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from s2t.paths import PROJECT_ROOT
from s2t.workflows.runner import run_cmd
from s2t.evaluation import run_baseline_benchmarks as baseline


def test_runner_uses_repository_root_and_stops_on_failure():
    command = ['python', '-m', 's2t.training.train_lora']
    with patch('s2t.workflows.runner.subprocess.run', return_value=SimpleNamespace(returncode=2)) as run:
        with pytest.raises(RuntimeError, match='training.*exit code 2'):
            run_cmd(command, 'training')
        run.assert_called_once_with(command, cwd=PROJECT_ROOT)
    assert PROJECT_ROOT == Path(__file__).resolve().parents[1]


def test_baseline_delegates_and_preserves_group_summaries():
    result = {
        'wer': 12.5, 'wer_splitmerge': 10.0, 'gender_results': {},
        'samples': [
            {'speaker': 'HQTV', 'ref_norm': 'hello', 'hyp_norm': 'hello'},
            {'speaker': 'PNV', 'ref_norm': 'world', 'hyp_norm': 'world'},
        ],
    }
    with patch.object(baseline, 'evaluate_model_on_split', return_value=result) as evaluate, \
         patch.object(baseline, 'corpus_wer', return_value=0.0), \
         patch.object(baseline, 'corpus_cer', return_value=0.0):
        actual = baseline.evaluate_model_for_accent('model', 'processor', 'test', 'vietnamese', 3, 'cpu')
    evaluate.assert_called_once_with(
        model='model', processor='processor', split_name='test',
        accent='vietnamese', batch_size=3, device='cpu',
    )
    assert actual['wer_splitmerge'] == 10.0
    assert actual['gender_results']['Male Avg']['n_utts'] == 1
    assert actual['gender_results']['Female Avg']['n_utts'] == 1


def test_publishing_import_does_not_authenticate():
    import importlib
    import sys
    with patch('huggingface_hub.HfApi') as api:
        sys.modules.pop('s2t.publishing.push_to_hf', None)
        publishing = importlib.import_module('s2t.publishing.push_to_hf')
        api.assert_not_called()
        card = publishing.create_accent_model_card(publishing.MODELS_SPEC[0], 'example-user')
    assert 'example-user/' in card
