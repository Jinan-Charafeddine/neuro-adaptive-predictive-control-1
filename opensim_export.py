"""Export author-supplied OpenSim model states. Native bindings required.
Not the unpublished closed-loop controller. Not tested in this environment.
"""
from pathlib import Path
import argparse, csv

def export(model_path, output, duration=1.0, dt=.01):
    import opensim as osim
    if not model_path.is_file():raise FileNotFoundError('Supply the original .osim model')
    model=osim.Model(str(model_path));state=model.initSystem();model.equilibrateMuscles(state)
    manager=osim.Manager(model);manager.initialize(state)
    coords=model.getCoordinateSet();muscles=model.getMuscles();output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['time_s','provenance']+[coords.get(i).getName()+'_rad' for i in range(coords.getSize())]+[muscles.get(i).getName()+'_activation' for i in range(muscles.getSize())])
        for k in range(int(duration/dt)+1):
            if k:state=manager.integrate(k*dt)
            model.realizeDynamics(state)
            writer.writerow([state.getTime(),'opensim_model_output']+[coords.get(i).getValue(state) for i in range(coords.getSize())]+[muscles.get(i).getActivation(state) for i in range(muscles.getSize())])
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('model',type=Path);p.add_argument('--output',type=Path,default=Path('results/opensim_states.csv'));a=p.parse_args();export(a.model,a.output)
