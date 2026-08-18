"""
Tests pour le pipeline Kedro
"""
import pytest
from pathlib import Path
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project


class TestKedroRun:
    def test_kedro_run(self):
        # Vérifier que kedro run s'exécute sans erreur
        bootstrap_project(Path.cwd())
        with KedroSession.create(project_path=Path.cwd()) as session:
            result = session.run()
        assert result is not None