import argparse
import os

DEFAULT_ENVS = [
    "AbandonedCable",
    "AbandonedFactory",
    "AbandonedSchool",
    "AmericanDiner",
    "AmusementPark",
    "AncientTowns",
    "Antiquity3D",
    "BrushifyMoon",
    "CarWelding",
    "CastleFortress",
    "CoalMine",
    "ConstructionSite",
    "CountryHouse",
    "Cyberpunk",
    "CyberPunkDowntown",
    "DesertGasStation",
    "Downtown",
    "EndofTheWorld",
    "FactoryWeather",
    "Fantasy",
    "ForestEnv",
    "Gascola",
    "GothicIsland",
    "GreatMarsh",
    "HongKong",
    "Hospital",
    "IndustrialHangar",
    "JapaneseAlley",
    "JapaneseCity",
    "MiddleEast",
    "ModernCityDowntown",
    "ModularNeighborhood",
    "ModularNeighborhoodIntExt",
    "ModUrbanCity",
    "NordicHarbor",
    "Ocean",
    "OldBrickHouseDay",
    "OldBrickHouseNight",
    "OldIndustrialCity",
    "OldScandinavia",
    "OldTownFall",
    "OldTownNight",
    "OldTownSummer",
    "OldTownWinter",
    "PolarSciFi",
    "Prison",
    "Rome",
    "Ruins",
    "SeasideTown",
    "SeasonalForestAutumn",
    "SeasonalForestSpring",
    "SeasonalForestSummerNight",
    "SeasonalForestWinter",
    "SeasonalForestWinterNight",
    "ShoreCaves",
    "TerrainBlending",
    "UrbanConstruction",
    "VictorianStreet",
    "WaterMillDay",
    "WaterMillNight",
    "WesternDesertTown",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Download TartanAir dataset.")
    parser.add_argument(
        "--data-root",
        default=os.environ.get("TARTANAIR_DATA_ROOT", "raw"),
        help="Directory to store downloaded data (default: raw, or $TARTANAIR_DATA_ROOT)",
    )
    parser.add_argument(
        "--data-source",
        choices=["airlab", "huggingface"],
        default="huggingface",
        help="Download source (default: airlab)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of parallel download workers (default: 4)",
    )
    parser.add_argument(
        "--env",
        nargs="+",
        default=DEFAULT_ENVS,
        help=f"Environment name(s) to download (default: all {len(DEFAULT_ENVS)} envs)",
    )
    parser.add_argument(
        "--difficulty",
        nargs="+",
        default=["hard"],
        choices=["easy", "hard"],
        help="Difficulty level(s) to download (default: hard)",
    )
    parser.add_argument(
        "--modality",
        nargs="+",
        default=["image", "depth"],
        help="Modality(s) to download, e.g. image depth seg flow (default: image depth)",
    )
    parser.add_argument(
        "--camera-name",
        nargs="+",
        default=[
            "lcam_front",
            "lcam_right",
            "lcam_back",
            "lcam_left",
            "lcam_top",
            "lcam_bottom",
        ],
        help="Camera name(s) to download (default: all lcam except fish/equirect)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    import cv2  # must import before tartanair to avoid libtiff/libjpeg conflict
    import tartanair as ta

    data_root = os.path.abspath(args.data_root)
    os.makedirs(data_root, exist_ok=True)

    print(f"Download path: {data_root}")
    print(f"env: {args.env}")
    print(f"difficulty: {args.difficulty}")
    print(f"modality: {args.modality}")
    print(f"camera_name: {args.camera_name}")
    ta.init(data_root)

    ta.download(
        env=args.env,
        difficulty=args.difficulty,
        modality=args.modality,
        camera_name=args.camera_name,
        unzip=True,
        num_workers=args.num_workers,
        data_source=args.data_source,
    )


if __name__ == "__main__":
    main()
