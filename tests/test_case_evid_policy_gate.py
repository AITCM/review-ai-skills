"""Synthetic regression tests for CASE-EVID research SOP gate."""
import copy,json,unittest
from pathlib import Path
import importlib.util
P=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("policy_gate",P/"tools"/"case_evid_policy_gate.py")
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
class GateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy=json.loads((P/"configs"/"case_evid_policy_v1.json").read_text())
        cls.example=json.loads((P/"configs"/"experiments"/"E083_N10_v1.json").read_text())
    def test_valid(self):
        self.assertEqual(mod.check_experiment(self.policy,self.example,"print('ok')"),[])
    def test_forbid_test(self):
        self.assertTrue(mod.check_experiment(self.policy,self.example,"path='test-00000-of-00001.parquet'"))
    def test_no_query_outcome(self):
        e=copy.deepcopy(self.example);e["data"]["query_fields"].append("diagnostic_reasoning")
        self.assertTrue(mod.check_experiment(self.policy,e,"x=1"))
    def test_no_medr_tuning(self):
        e=copy.deepcopy(self.example);e["governance"]["medr_not_used_for_tuning"]=False
        self.assertTrue(mod.check_experiment(self.policy,e,"x=1"))
    def test_group_split(self):
        e=copy.deepcopy(self.example);e["data"]["grouped_split"]=False
        self.assertTrue(mod.check_experiment(self.policy,e,"x=1"))
    def test_no_false_confirmation(self):
        e=copy.deepcopy(self.example);e["status"]="independent_external"
        self.assertTrue(mod.check_experiment(self.policy,e,"x=1"))
    def test_bad_python(self):
        self.assertTrue(mod.check_experiment(self.policy,self.example,"def invalid("))
if __name__=="__main__":
    unittest.main()
