"""Apply observation instrumentation and native-outcome recording to the client."""
import pathlib,os,sys
C=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(C))

p=C/'RoboDojo/src/eval_client/main.py'
source=p.read_text()
needle='from src.eval_client.eval_env import create_eval_env'
assert source.count(needle)==1
addition='\nfrom measure import instrument_factory\ncreate_eval_env = instrument_factory(create_eval_env)'

source=source.replace(needle,needle+addition)
scope={'__name__':'__main__','__file__':str(p),'__package__':None}
try:
    exec(compile(source,str(p),'exec'),scope)
except BaseException:
    import traceback
    traceback.print_exc();sys.stdout.flush();sys.stderr.flush();os._exit(1)
