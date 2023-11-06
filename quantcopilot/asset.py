import quantframe as qf
from quantframe.database import DatabaseBase


class Asset(qf.AssetBase):

    def __init__(
        self, 
        code: str, 
        database: DatabaseBase = None
    ) -> None:
        super().__init__(code, database)
        self.universe_type = Universe


class Universe(qf.UniverseBase):

    def __init__(
        self, 
        *assets: list[str] | list[Asset],
        database: DatabaseBase = None
    ) -> None:
        self.assets = []
        self.codes = []
        for asset in assets:
            if isinstance(asset, str):
                self.assets.append(Asset(asset, database))
                self.codes.append(asset)
            elif isinstance(asset, Asset):
                self.assets.append(asset)
                self.codes.append(asset.code)
