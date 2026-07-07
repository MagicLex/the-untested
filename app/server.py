"""the-untested app server. Thin FastAPI: serves the static canvas frontend and
the map data live from HopsFS, draws any molecule with RDKit, and proxies the
amrscorer endpoint. Front-end edits go live on refresh (files read per request);
only server code changes need a restart.
"""
import os

import hopsworks
from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from pydantic import BaseModel

APP_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(APP_DIR, "static")
app = FastAPI()
_dep = None


def _deployment():
    global _dep
    if _dep is None:
        proj = hopsworks.login()
        _dep = proj.get_model_serving().get_deployment("amrscorer")
    return _dep


@app.get("/", response_class=HTMLResponse)
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))


@app.get("/static/{name}")
def static_file(name: str):
    path = os.path.join(STATIC, name)
    if not os.path.isfile(path):
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(path)


@app.get("/api/depict")
def depict(smiles: str):
    from rdkit import Chem
    from rdkit.Chem.Draw import rdMolDraw2D
    m = Chem.MolFromSmiles(smiles or "")
    if m is None:
        return Response("<svg xmlns='http://www.w3.org/2000/svg'/>",
                        media_type="image/svg+xml")
    d = rdMolDraw2D.MolDraw2DSVG(320, 240)
    o = d.drawOptions()
    o.setBackgroundColour((0.055, 0.067, 0.09, 0.0))
    o.bondLineWidth = 2
    rdMolDraw2D.PrepareAndDrawMolecule(d, m)
    d.FinishDrawing()
    svg = d.GetDrawingText().replace("#000000", "#d7e0ea")
    return Response(svg, media_type="image/svg+xml")


class ScoreReq(BaseModel):
    smiles: str


@app.post("/api/score")
def score(req: ScoreReq):
    try:
        dep = _deployment()
        if not dep.is_running():
            return JSONResponse({"error": "endpoint starting, try again shortly"},
                                status_code=503)
        return dep.predict(inputs=[{"smiles": req.smiles}])
    except Exception as e:
        return JSONResponse({"error": str(e)[:200]}, status_code=500)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("APP_PORT", "8080")))
