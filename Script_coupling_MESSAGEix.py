import numpy as np
import pandas as pd
import yaml
from yaml.loader import SafeLoader

from message_ix_models.tools.add_dac import add_tech

from message_ix_models.util import private_data_path, broadcast
from message_data.tools.utilities.get_historical_years import (
    main as get_historical_years,
)
from message_data.tools.utilities.get_optimization_years import (
    main as get_optimization_years,
)
from message_ix.util import make_df


ship_types_mariteam = [
    "bulk",  # bulk carriers
    # "chemical",  # chemical tankers
    "container",  # container ships
    "general",  # general cargo ships
    # "LNG",  # liquefied gas carriers
    # "oil",  # oil tankers
    "roro",  # ro-ro ships
]
haul_types = ["short", "long"]
cohort_types = ["current", "future"]
# List of emissions (currently excludes "CO2", "NH3"
species_list = ["CH4", "N2O", "VOC", "NOx", "CO", "BCA", "SO2", "OCA"]
fuel_list = [
    "foil_bunker",
    "loil_bunker",
    "LNG_bunker",
    "meth_bunker",
    "eth_bunker",
    "LH2_bunker",
    "NH3_bunker",
]
occ_list = [
    "foil_occ_bunker",
    "loil_occ_bunker",
    "LNG_occ_bunker",
]

# Historic projections based on defaults historic data from IMO.
# The annual factors are always applied to the 2020 value.
historic_reference_year = 2020
historic_factors = {
    1990: 0.547,
    1995: 0.648,
    2000: 0.773,
    2005: 0.946,
    2010: 0.909,
    2015: 0.936,
    2020: 1.0,
    2025: 1.0,
}


class MariteamShipping:
    def __init__(
        self,
        mp,
        scen,
        base,
        par_fil="parameters_SSPs.yaml",
        ssp="SSP2",
        verbose=False,
    ):
        """
        Apply MariTEAM shipping implementation.

        This script will modify a scenario and update the
        "shipping-representation" in MESSAGE at the global level.

        Parameters
        ----------
        scen :
            Scenario to which changes should be applied.
        base :
            Scenario that provides base energy demand for first year
            of the model.
        par_fil : str
            Name of file, incl. extension, containing configuration parameters.
        ssp : str  (defaul="SSP2")
            Name of SSP for which the data is being processed.

        """

        self.mp = mp
        self.scen = scen
        self.base = base if not None else self.scen
        self.data_path = private_data_path("mariteam_shipping")
        self.ssp = ssp
        self.verbose = verbose

        # ---------------------------
        # Specify source/output files
        # ---------------------------

        # Step1. Data compilation related parameters
        # Specify name of file generated as part of Step-1.
        self.export_message_data_file = self.data_path / "mariteam_compiled.csv"

        # Emission factors
        self.emission_factors_file = self.data_path / "emission_factors.csv"

        # Change this to search autmaticaly in scenario
        # Derive the region where the technologies for shipping demand are added
        self.reg = [n for n in self.scen.set("node") if "GLB" in n][0]
        # Retrieve historic years
        self.year_hist = get_historical_years(scen)
        # Retrieve optimization time-periods
        self.year_opt = get_optimization_years(scen)
        # Retrieve all model periods
        self.years = self.year_hist + self.year_opt

        # Retrieve input parameters
        self.retrieve_input_parameters(par_fil)

        # ----------------------------
        # Step 1. Import MariTEAM data with SSP projections
        # ----------------------------
        print("Importing MariTeam SSP data") if self.verbose == True else None

        self.export_message = pd.read_csv(self.export_message_data_file, index_col=0)

        # ----------------------------
        # Step 2. Add Shipping in the scenario
        # ----------------------------
        print("Adding new shipping in the scenario") if self.verbose == True else None

        with self.scen.transact("Apply Shipping"):
            # Add MariTeam demand
            self.add_mariteam_scen()

            # Add energy carrier sectors
            self.add_shp_energy()

            # -------------------------
            # Step 3. Add fuels and connect to shipping
            # -------------------------
            (
                print("Add novel fuels and abatement tech")
                if self.verbose == True
                else None
            )

            # Add ammonia as a fuel and synthetic e-fuels
            self.add_NH3_bunker()
            # self.add_synt_foil()

            # Add Onboard Carbon Capture (OCC)
            self.add_OCC_shp()
            self.connect_CO2_storage()

            # -------------------------
            # Step 4. Add fuel pathways
            # -------------------------
            (
                print("Connect new shipping modes to fuels")
                if self.verbose == True
                else None
            )

            # Add bunker fuels to supply shipping
            self.add_fuel_pathways()

            # Add constraints for the ship transition
            self.add_fuel_deployment_constraints()

            # -------------------------
            # Step 5. Add emissions
            # -------------------------
            print("Update emissions") if self.verbose == True else None

            # Add emission factors of bunker fuels
            self.add_fuels_EF_new()

            # -------------------------
            # Step 6. Model first year fuel mix
            # -------------------------
            print("Constraints for fuel mix") if self.verbose == True else None

            # Fuel mix in the first year and IMO goals
            self.baseline_2030()
            self.IMO_goal()

            # Remove all fuels persisting in M1 mode (to be fixed)
            self.remove_old_fuel_pathways()

    def retrieve_input_parameters(self, parameter_file):
        """
        Retrieves input parameters from a YAML file.

        This method reads input parameters from a specified
        YAML file and stores them in the class attribute input_parameters.
        """

        infil = self.data_path / parameter_file  # YAML file path
        with open(infil) as f:
            self.input_parameters = yaml.load(f, Loader=SafeLoader)

    def add_mariteam_scen(self):
        """
        Assign shipping demands based on MariTEam demands.

        This method adds shipping demand for various sectors based on values
        obtained from the MariTeam model. It iterates through different ship
        types, haul types, and cohort types to assign demands accordingly.
        The shipping demands are adjusted for bulk trade by applying a
        factor of 0.65 to reflect the portion related to coal and steel.
        Historic projections are added based on default historic data,
        with relative adjustments compared to the year 2015.
        """

        # get demands from the MariTeam model
        mariteam_demand = self.export_message
        mariteam_demand = mariteam_demand.loc[mariteam_demand["scenario"] == self.ssp]
        mariteam_demand.reset_index(inplace=True)
        mariteam_demand = mariteam_demand.loc[
            mariteam_demand["year"] >= self.scen.firstmodelyear
        ]

        # demand for shipping from the baseline scenario
        base_df = self.base.var("DEMAND", filters={"commodity": "shipping"})
        base_df = base_df.loc[base_df["year"] >= self.scen.firstmodelyear]

        # create shipping commodities/modes and assign demand
        shp_sec = {}
        for ship_type in ship_types_mariteam:
            for distance in haul_types:
                for cohort in cohort_types:
                    sector = f"shp_{ship_type}_{distance}_{cohort}"
                    mode = f"shp_{ship_type}_{distance}_{cohort}"
                    shp_sec[sector] = mode
                    self.scen.add_set("mode", mode)
                    self.scen.add_set("commodity", [sector])
                    tmp = (
                        base_df.loc[base_df.year.isin(mariteam_demand.year)]
                        .copy()
                        .set_index("year")
                    )
                    # The 65% value reflects the fact that 25% of the bulk
                    # trade is actually related to coal 10% steel.
                    # Values from UNCTAD's Review of Maritime Transport 2019.
                    value_factor = 0.65 if ship_type == "bulk" else 1
                    tmp["value"] = (
                        mariteam_demand[[sector, "year"]].set_index("year")
                        * value_factor
                    )
                    tmp = tmp.assign(
                        commodity=sector,
                        mode=mode,
                        unit="GWa",
                        level="useful",
                    ).reset_index()

                    # self.scen.add_par(
                    #    "demand",
                    #    base_df.loc[base_df.year.isin(mariteam_demand.year)].assign(
                    #        commodity=sector,
                    #        mode=mode,
                    #        value=list( mariteam_demand[sector] * np.where(ship_type == "bulk", 0.65, 1)),
                    #        unit="GWa",
                    #        level="useful",
                    #    ),
                    # )
                    self.scen.add_par("demand", tmp)
        self.ship_modes = shp_sec

        # The sectors treated exclude any "future" demand trajectories, therefore
        # only applicable to the "current" demands.
        for sector in shp_sec.keys():
            hist_shp_dem = self.scen.par("demand", filters={"commodity": sector})
            for y in historic_factors.keys():
                if y in mariteam_demand.year.tolist():
                    continue
                tmp = hist_shp_dem.loc[
                    hist_shp_dem.year == historic_reference_year
                ].assign(year=y)
                if "current" in sector:
                    tmp.value *= historic_factors[y]
                else:
                    tmp.value = 0
                hist_shp_dem = pd.concat([hist_shp_dem, tmp], ignore_index=True)
            self.scen.add_par("demand", hist_shp_dem)

    #            self.scen.add_par(
    #                "demand",
    #                base_df.iloc[-5:].assign(
    #                    commodity=sector,
    #                    mode=shp_sec[sector],
    #                    value=list(
    #                        np.array([0.93, 0.93, 0.9, 1.0, 1.0])
    #                        * mariteam_demand[sector].iloc[0]
    #                    ),
    #                    unit="GWa",
    #                    year=self.year_hist[-5:],
    #                    level="useful",
    #                ),
    #            )

    def add_shp_energy(self):
        """
        Adds energy carriers as feedback from fossil fuels and commodities.

        This method adds energy carriers based on the demand for fossil fuels
        and commodities. It calculates activity levels in 2020 and sets the
        corresponding inputs for various energy sectors such as oil, LNG,
        solid fuels, and chemicals.
        """

        # Add energy carriers as feedback from fossil fuels and commodities
        energy_sectors = [
            "shp_oil",
            # "shp_LNG", # currently not working properly due to LNG_trd
            "shp_solid",
            "shp_chemical",
        ]
        energy_sectors_mode = []
        ship_modes = self.ship_modes.copy()
        for sector in energy_sectors:
            self.scen.add_set("commodity", sector)
            self.scen.add_set("mode", sector)
            energy_sectors_mode.append(sector)
            ship_modes[sector] = sector
        self.ship_modes = ship_modes
        self.energy_sectors = energy_sectors
        energy_sectors_firstyear = []

        # Get activity levels in the first year of the model
        shp_demand_firstyear = self.export_message
        shp_demand_firstyear = shp_demand_firstyear[
            (shp_demand_firstyear["year"] == self.scen.firstmodelyear)
            & (shp_demand_firstyear["scenario"] == self.ssp)
        ]
        baseline_firstyear = self.base.var(
            "ACT",
            filters={
                "year_act": self.scen.firstmodelyear,
                "technology": [
                    "oil_trd",
                    "loil_trd",
                    "eth_trd",
                    "trade_NH3",
                    "trade_petro",
                    "coal_trd",
                    "trade_steel",
                    "biomass_trd",
                ],
            },
        )
        base_df = {
            "node_loc": self.reg,
            "node_origin": self.reg,
            "node_dest": self.reg,
            "year_vtg": self.year_opt,
            "year_act": self.year_opt,
            "time": "year",
            "time_origin": "year",
            "time_dest": "year",
            "unit": "-",
            "mode": "M1",
            "level": "useful",
        }
        hist_demand = {}
        index_hist_demand = 0

        # Oil trade based on oil_trd and loil_trd
        if "shp_oil" in energy_sectors:
            tec_list = ["oil_trd", "loil_trd"]
            act_mariteam_firstyear = sum(
                [
                    shp_demand_firstyear[f"shp_oil_{distance}_current"].iloc[0]
                    for distance in haul_types
                ]
            )
            for tec, share in zip(tec_list, [0.83, 0.17]):
                act_message_firstyear = baseline_firstyear.loc[
                    (baseline_firstyear["technology"] == tec)
                ]["lvl"].iloc[0]
                inputdf = make_df(
                    base_df,
                    technology=tec,
                    commodity="shp_oil",
                    value=act_mariteam_firstyear * share / act_message_firstyear,
                )
                self.scen.add_par("input", inputdf)
                energy_sectors_firstyear.append(act_mariteam_firstyear)
        hist_demand["shp_oil"] = act_mariteam_firstyear

        # Chemical trade
        if "shp_chemical" in energy_sectors:
            tec_list = ["eth_trd", "trade_NH3", "trade_petro"]
            act_mariteam_firstyear = sum(
                [
                    shp_demand_firstyear[f"shp_chemical_{distance}_current"].iloc[0]
                    for distance in haul_types
                ]
            )
            # little fix because eth_trd is 0 in in the first year
            for tec, tec_filter, share in zip(
                tec_list,
                ["eth_trd", "trade_NH3", "trade_petro"],
                [0.4, 0.2, 0.4],  # [0.3, 0.25, 0.65]
            ):
                act_message_firstyear = baseline_firstyear.loc[
                    (baseline_firstyear["technology"] == tec_filter)
                ]["lvl"].iloc[0]
                inputdf = make_df(
                    base_df,
                    technology=tec,
                    commodity="shp_chemical",
                    value=act_mariteam_firstyear * share / act_message_firstyear,
                )
                self.scen.add_par("input", inputdf)
                energy_sectors_firstyear.append(act_mariteam_firstyear)
        hist_demand["shp_chemical"] = act_mariteam_firstyear

        # Bulk trade
        if "shp_solid" in energy_sectors:
            tec_list = ["coal_trd", "trade_steel", "biomass_trd"]
            # energy density of coal is 24MJ/kg and wood is 16, so we need 50%
            # more wood to be transported and thus 50% more shipping
            adjust_coal_biomass = 24 / 16
            # energy density of coal is 24MJ/kg and wood is 16, so we need 50% more
            # wood to be transported and thus 50% more shipping
            share_current_bulk = [
                0.25,
                0.1,
                0.25 * adjust_coal_biomass,
            ]  # Values from UNCTAD's Review of Maritime Transport 2019
            act_mariteam_firstyear = sum(
                [
                    shp_demand_firstyear[f"shp_bulk_{distance}_current"].iloc[0]
                    for distance in haul_types
                ]
            )
            for i, tec in enumerate(tec_list):
                act_message_firstyear = baseline_firstyear.loc[
                    (baseline_firstyear["technology"] == tec)
                ]["lvl"].iloc[0]
                inputdf = make_df(
                    base_df,
                    technology=tec,
                    commodity="shp_solid",
                    value=act_mariteam_firstyear
                    * share_current_bulk[i]
                    / act_message_firstyear,
                )
                self.scen.add_par("input", inputdf)
        hist_demand["shp_solid"] = act_mariteam_firstyear

        # LNG trade
        if "shp_LNG" in energy_sectors:
            tec_list = ["LNG_trd"]
            act_mariteam_firstyear = sum(
                [
                    shp_demand_firstyear[f"shp_LNG_{distance}_current"].iloc[0]
                    for distance in haul_types
                ]
            )
            for tec in tec_list:
                act_message_firstyear = baseline_firstyear.loc[
                    (baseline_firstyear["technology"] == tec)
                ]["lvl"].iloc[0]
                inputdf = make_df(
                    base_df,
                    technology=tec,
                    commodity="shp_LNG",
                    value=act_mariteam_firstyear / act_message_firstyear,
                )
                self.scen.add_par("input", inputdf)
                energy_sectors_firstyear.append(act_mariteam_firstyear)

        # Set energy trade back to 1 so prevent double account of transp
        # This is to remove the efficiencies which were previously set to account for
        # shipping losses, which will now be handled directly by the shipping implementation.
        for tec in [
            "oil_trd",
            "loil_trd",
            "foil_trd",
            "LNG_trd",
            "coal_trd",
            "meth_trd",
            "eth_trd",
            "trade_NH3",
            "trade_petro",
        ]:
            filter_param = self.scen.par(
                "input", filters={"technology": tec, "level": "export"}
            )
            filter_param["value"] = 1
            self.scen.add_par("input", filter_param)

        for sector in energy_sectors:
            hist_data = pd.DataFrame(
                {
                    "node": "R12_GLB",
                    "commodity": sector,
                    "level": "useful",
                    "year": [
                        y
                        for y in historic_factors.keys()
                        if y < self.scen.firstmodelyear
                    ],
                    "time": "year",
                    "value": [
                        historic_factors[y] * hist_demand[sector]
                        for y in historic_factors.keys()
                        if y < self.scen.firstmodelyear
                    ],
                    "unit": "GWa",
                }
            )
            self.scen.add_par("demand", hist_data)

        #        # Historic projections based on defaults historic data from IEA.
        #            base_df = self.base.var("DEMAND", filters={"commodity": "shipping"})
        #            self.scen.add_par(
        #                "demand",
        #                base_df.iloc[-5:].assign(
        #                    commodity=sector,
        #                    mode=sector,
        #                    value=list(np.array([0.93, 0.93, 0.9, 1.0, 1.0]) * hist_demand[i]),
        #                    unit="GWa",
        #                    year=self.year_hist[-5:],
        #                    level="useful",
        #                ),
        #            )

        # Remove existing shipping demand
        df = self.scen.par("demand", filters={"commodity": "shipping"})
        self.scen.remove_par("demand", df)

    def add_NH3_bunker(self):
        """
        Adds new technology NH3_bunker and NH3_tobunker.

        This method adds new technology NH3_bunker and NH3_tobunker and sets the
        corresponding inputs and outputs. It converts ammonia from million tonnes
        to GWyr and adds input and output parameters accordingly.
        """

        # Add new technology
        self.scen.add_set("technology", ["NH3_bunker", "NH3_tobunker"])

        # Ammonia convert from million tonnes to GWyr
        # energy content = 18.8 MJ/kg
        convert_nh3 = 18.8  # 1 kg = 18.8MJ
        convert_nh3 *= 1e9  # 1 million tonne = convert MJ
        convert_nh3 *= 2.7777777777778e-7  # convert to GWh
        convert_nh3 /= 8765.81277  # convert to GWa

        base_df = {
            "node_loc": self.reg,
            "year_vtg": self.year_opt,
            "year_act": self.year_opt,
            "mode": "M1",
            "time": "year",
            "unit": "GWa",
        }
        base_input = make_df(base_df, node_origin=self.reg, time_origin="year")
        base_output = make_df(base_df, node_dest=self.reg, time_dest="year")
        fuel_in = make_df(
            base_input,
            technology="NH3_tobunker",
            commodity="NH3",
            level="export",
            value=1.0,
        )
        fuel_out = make_df(
            base_output,
            technology="NH3_tobunker",
            commodity="NH3",
            level="final",
            value=1 / convert_nh3,
        )
        self.scen.add_par("input", fuel_in)
        self.scen.add_par("output", fuel_out)

    def add_synt_foil(self):
        """
        Adds synthetic-diesel technology (loil_h2) with specific parameters.

        This method adds synthetic-diesel technology (loil_h2) with specific
        parameters such as input, output, fix_cost, inv_cost, and relation_activity.
        """

        # 5.2 Synthetic-diesel (only made from coal, adding one from H2)
        # - 'var_cost': it is all zero
        # - 'input': for meth_h2 1.115 of hydrogen and 0.0172 GWa of elec
        # - 'output': 1GWa methanol
        # - 'relation_activity': more complicated
        # according to this paper: https://ars.els-cdn.com/content/image/1-s2.0-S0360319919318580-mmc1.pdf
        mjkg_methanol = 22
        mjkg_diesel = 42.8
        ratio_kg = mjkg_methanol / mjkg_diesel
        input_h2 = (
            0.480 / 0.189 * ratio_kg * 1.11483
        )  # from publication, publication, message
        input_elec = 1.091 / 0.556 * ratio_kg * 0.0171613

        increase_fix_cost = 666 / 235  # from publication
        increase_inv_cost = 2.30 / 1.89  # from publication

        # this part copies meth_h2 -> loil_h2
        # change this to use "copy_parameters"
        # then change the parameters for inv_cost;

        self.scen.add_set("technology", ["loil_h2"])
        mode_params_dict = {}
        paramList_mode = [
            x for x in self.scen.par_list() if "mode" in self.scen.idx_sets(x)
        ]
        for p in paramList_mode:
            filter_param = self.scen.par(p, filters={"technology": "meth_h2"})
            mode_params_dict[p] = filter_param
            if p == "input":
                filter_param = filter_param.loc[filter_param["mode"] == "fuel"].copy()
                filter_param["technology"] = "loil_h2"
                filter_param_elec = filter_param.loc[
                    filter_param["commodity"] == "electr"
                ].copy()
                filter_param_elec["value"] = input_elec
                self.scen.add_par(p, filter_param_elec)
                filter_param_h2 = filter_param.loc[
                    filter_param["commodity"] == "hydrogen"
                ].copy()
                filter_param_h2["value"] = input_h2
                self.scen.add_par(p, filter_param_h2)
            elif p == "output":
                filter_param = filter_param.loc[
                    filter_param["commodity"] == "methanol"
                ].copy()
                filter_param = filter_param.loc[filter_param["mode"] == "fuel"]
                filter_param["commodity"] = "lightoil"
                filter_param["technology"] = "loil_h2"
                self.scen.add_par(p, filter_param)
            elif p == "fix_cost":
                filter_param["technology"] = "loil_h2"
                filter_param_h2["value"] *= increase_fix_cost
                self.scen.add_par(p, filter_param)
            elif p == "inv_cost":
                filter_param["technology"] = "loil_h2"
                filter_param_h2["value"] *= increase_inv_cost
                self.scen.add_par(p, filter_param)
            elif p == "relation_activity":
                filter_param = filter_param.loc[filter_param["mode"] == "fuel"].copy()
                filter_param["relation"] = "loil_h2"
                # CO2 emission
                filter_param_new = filter_param.loc[
                    filter_param["relation"] == "CO2_Emission"
                ]
                filter_param_new["value"] = 0.631
                self.scen.add_par(p, filter_param_new)
                # CO2_PtX_trans_disp_split
                filter_param_new = filter_param.loc[
                    filter_param["relation"] == "CO2_PtX_trans_disp_split"
                ]
                filter_param_new["value"] = -0.631
                self.scen.add_par(p, filter_param_new)
                # CO2_cc for accounting thing
                filter_param_new = filter_param.loc[filter_param["mode"] == "fuel"]
                filter_param_new["relation"] = "CO2_cc"
                filter_param_new["technology"] = "loil_h2"
                filter_param_new["value"] = 0.631
                self.scen.add_par(p, filter_param_new)

    def add_OCC_shp(self):
        """
        Adds operational carbon capture (OCC) technologies.

        This method adds operational carbon capture (OCC) technologies
        (foil_occ_bunker and loil_occ_bunker) and sets the corresponding inputs,
        outputs, and economic parameters based on Maritime Forecast 2050 by DNV.
        """

        # OCCs from Maritime Forecast 2050 by DNV
        fuel_penalty = {  # upper end
            "foil_occ_bunker": self.input_parameters["carbon_capture"][self.ssp][
                "fuel_penalty"
            ],
            "loil_occ_bunker": self.input_parameters["carbon_capture"][self.ssp][
                "fuel_penalty"
            ],
            "LNG_occ_bunker": self.input_parameters["carbon_capture"][self.ssp][
                "fuel_penalty"
            ],
        }
        # cc_rate = 0.7 #self.input_parameters["carbon_capture"]["high"]["cc_rate"]

        # From the same report, one study case of a container ships
        energy_demand_ship_year = 24500  # tonnes of VLSFO
        energy_demand_ship_year *= 42.5 * 1e3  # MJ/kg
        energy_demand_ship_year *= 2.7777777777778e-7  # from MJ to GWh
        energy_demand_ship_year /= 365.4 * 24  # from GWh to GW

        # Capex and Opex values
        capex_ship = 20e6  # dollars
        capex_add = (
            0.125  # 12.5% additional for scrubber, CO2 capture unit, and storage tanks.
        )
        opex_ship = 1e6  # dollars
        # carbon_content_fuel = 0.86 # 86% carbon
        capex_message_occ = (
            capex_ship
            * (1 + capex_add)
            / energy_demand_ship_year
            / self.input_parameters["carbon_capture"][self.ssp]["lifetime_occ"]
        )
        opex_message_occ = (
            opex_ship * 25 / energy_demand_ship_year
        )  # unit: dollars/GWa for 70% capture
        # multiplification by 5 includes the increase in costs to handle
        # and store the CO2 in ports. Based on DNV estimates of 65 to
        # 130 dollars per ton CO2.

        # Add OCC technologies
        for tec in ["foil_occ_bunker", "loil_occ_bunker", "LNG_occ_bunker"]:
            self.scen.add_set("technology", tec)

        # Add input/outputs to OCC
        for sector, mode in zip(self.ship_modes, self.ship_modes):
            base_output = {
                "node_loc": self.reg,
                "year_vtg": self.year_opt,
                "year_act": self.year_opt,
                "mode": mode,
                "node_origin": self.reg,
                "node_dest": self.reg,
                "time": "year",
                "time_dest": "year",
                "time_origin": "year",
            }
            po_out = make_df(
                base_output,
                technology="foil_occ_bunker",
                commodity=sector,
                level="useful",
                value=1 / fuel_penalty["foil_occ_bunker"],
                unit="-",
            )
            self.scen.add_par("output", po_out)
            po_out = make_df(
                base_output,
                technology="loil_occ_bunker",
                commodity=sector,
                level="useful",
                value=1 / fuel_penalty["loil_occ_bunker"],
                unit="-",
            )
            self.scen.add_par("output", po_out)
            po_out = make_df(
                base_output,
                technology="LNG_occ_bunker",
                commodity=sector,
                level="useful",
                value=1 / fuel_penalty["LNG_occ_bunker"],
                unit="-",
            )
            self.scen.add_par("output", po_out)

            base_input = {
                "node_loc": self.reg,
                "year_vtg": self.year_opt,
                "year_act": self.year_opt,
                "mode": mode,
                "node_origin": self.reg,
                "time": "year",
                "time_origin": "year",
            }
            po_in = make_df(
                base_input,
                technology="foil_occ_bunker",
                commodity="fueloil",
                level="final",
                value=1.0,
                unit="-",
            )
            self.scen.add_par("input", po_in)
            po_in = make_df(
                base_input,
                technology="loil_occ_bunker",
                commodity="lightoil",
                level="final",
                value=1.0,
                unit="-",
            )
            self.scen.add_par("input", po_in)
            po_in = make_df(
                base_input,
                technology="LNG_occ_bunker",
                commodity="LNG",
                level="final",
                value=1.0,
                unit="-",
            )
            self.scen.add_par("input", po_in)

        # Add economic parameters
        for tec in ["foil_occ_bunker", "loil_occ_bunker", "LNG_occ_bunker"]:
            # technical lifetime
            base_technical_lifetime = {
                "node_loc": self.reg,
                "year_vtg": self.year_opt,
                "unit": "y",
            }
            po_tl = make_df(
                base_technical_lifetime,
                technology=tec,
                value=self.input_parameters["carbon_capture"][self.ssp][
                    "lifetime_occ"
                ],  # same as ship lifetime
            )
            self.scen.add_par("technical_lifetime", po_tl)

            # investment cost
            base_inv_cost = {
                "node_loc": self.reg,
                "year_vtg": self.year_opt,
                "unit": "USD/kW",
            }
            po_inv = make_df(
                base_inv_cost,
                technology=tec,
                value=capex_message_occ / 1e6,  # from GW to kw
            )
            self.scen.add_par("inv_cost", po_inv)

            # base fix cost
            for year in self.year_opt:
                base_fix_cost = {
                    "node_loc": self.reg,
                    "year_vtg": year,
                    "year_act": self.year_opt,
                    "unit": "USD/kW",
                }
                po_fix = make_df(
                    base_fix_cost,
                    technology=tec,
                    value=opex_message_occ / 1e6,  # from GW to kw
                )
                self.scen.add_par("fix_cost", po_fix)

        # Add add-on technologies
        type_addon_ch = "occ_foil"
        self.scen.add_set("addon", "foil_occ_bunker")
        self.scen.add_cat("addon", type_addon_ch, "foil_occ_bunker")
        self.scen.add_set(
            "map_tec_addon",
            pd.DataFrame({"technology": "foil_bunker", "type_addon": [type_addon_ch]}),
        )
        type_addon_ch = "occ_loil"
        self.scen.add_set("addon", "loil_occ_bunker")
        self.scen.add_cat("addon", type_addon_ch, "loil_occ_bunker")
        self.scen.add_set(
            "map_tec_addon",
            pd.DataFrame({"technology": "loil_bunker", "type_addon": [type_addon_ch]}),
        )
        type_addon_ch = "occ_LNG"
        self.scen.add_set("addon", "LNG_occ_bunker")
        self.scen.add_cat("addon", type_addon_ch, "LNG_occ_bunker")
        self.scen.add_set(
            "map_tec_addon",
            pd.DataFrame({"technology": "LNG_bunker", "type_addon": [type_addon_ch]}),
        )

        # Add other paramerters
        for mode in self.ship_modes.values():
            for tec, type_addon_ch in zip(
                ["foil_bunker", "loil_bunker", "LNG_bunker"],
                ["occ_foil", "occ_loil", "occ_LNG"],
            ):
                df = pd.DataFrame(
                    {
                        "node": self.reg,
                        "technology": tec,
                        "year_vtg": self.year_opt,
                        "year_act": self.year_opt,
                        "mode": mode,
                        "time": "year",
                        "type_addon": type_addon_ch,
                        "value": 1.0,  # if it makes any conversion of units/fuel penalty
                        "unit": "-",
                    }
                )
                self.scen.add_par("addon_conversion", df)
            for tec, type_addon_ch in zip(
                ["foil_bunker", "loil_bunker", "LNG_bunker"],
                ["occ_foil", "occ_loil", "occ_LNG"],
            ):
                df = pd.DataFrame(
                    {
                        "node": self.reg,
                        "technology": tec,
                        "year_act": self.year_opt,
                        "mode": mode,
                        "time": "year",
                        "type_addon": type_addon_ch,
                        "value": 1.0,  # 100% of ships can run with OCC
                        "unit": "-",
                    }
                )
                self.scen.add_par("addon_up", df)

    def add_fuels_EF(self):
        """
        DEPRECATED - replaced by add_fuels_EF_new
        TODO: remove this method once add_fuels_EF_new is confirmed to work as expected

        Add emission factors for different bunker fuels.

        This method adds emission factors for different fuels and improves ammonia
        emission factors across the years. It also adds emission factors for operational
        carbon capture (OCC) technologies.
        Reference for emissions: DOI:10.3390/atmos14050879 (compiled in CSV file)
        """

        self.emission_factors = pd.read_csv(self.emission_factors_file, index_col=0)

        # Add shipping CO2 emissions in all modes
        relation_co2 = self.scen.par(
            "relation_activity",
            filters={"relation": "CO2_shipping"},
        )
        relation_co2 = relation_co2.loc[relation_co2["technology"].isin(fuel_list)]
        self.scen.remove_par("relation_activity", relation_co2)
        for mode in self.ship_modes:
            relation_co2["mode"] = mode
            self.scen.add_par("relation_activity", relation_co2)

        # Add emission factor for each fuel for non-CO2 species
        relation_emiss = self.scen.par(
            "relation_activity",
            filters={"relation": "CO2_shipping", "technology": "LNG_bunker"},
        )
        for emission in species_list:
            efs = self.emission_factors
            efs = efs.loc[efs["species"] == emission]
            relation_emiss["relation"] = f"{emission}_Emission_bunkers"
            for fuel in fuel_list:
                relation_emiss["technology"] = fuel
                efs_val = efs[fuel].iloc[0]
                relation_emiss["value"] = efs_val
                self.scen.add_par("relation_activity", relation_emiss)

    def add_fuels_EF_new(self):
        """
        Introduces new relations (<species>_Emission_bunkers) serving as
        emission factors for new bunker technology-mode permutations.

        It uses the values per technology from the literature source
        for non-CO2 species. Values are rounded to 7 digits for solver performance.
        The values are broadcasted to all technology modes and years
        assuming they will remain constant over time.

        CO2 emission factor values are kept as they are in MESSAGEix-GLOBIOM
        and only broadcasted to the new technology modes,
        while removing the deprecated M1 mode entries.

        Reference for emissions: DOI:10.3390/atmos14050879 (compiled in CSV file)
        """

        # Add shipping CO2 emissions in all modes
        relation_co2 = self.scen.par(
            "relation_activity",
            filters={"relation": "CO2_shipping"},
        )
        relation_co2 = relation_co2.loc[relation_co2["technology"].isin(fuel_list)]
        self.scen.remove_par("relation_activity", relation_co2)
        for mode in self.ship_modes:
            relation_co2["mode"] = mode
            self.scen.add_par("relation_activity", relation_co2)

        common = {
            "node_rel": "R12_GLB",
            "node_loc": "R12_GLB",
        }
        modes = [
            "shp_bulk_long_current",
            "shp_bulk_long_future",
            "shp_bulk_short_current",
            "shp_bulk_short_future",
            "shp_chemical",
            "shp_container_long_current",
            "shp_container_long_future",
            "shp_container_short_current",
            "shp_container_short_future",
            "shp_general_long_current",
            "shp_general_long_future",
            "shp_general_short_current",
            "shp_general_short_future",
            "shp_oil",
            "shp_roro_long_current",
            "shp_roro_long_future",
            "shp_roro_short_current",
            "shp_roro_short_future",
            "shp_solid",
        ]

        # read raw emission factor values and convert to long format
        # exclude zero values and CO2 entries from source file
        emi_raw = pd.read_csv(
            private_data_path("mariteam_shipping", "emission_factors.csv"), index_col=0
        )
        emi_raw = emi_raw.melt(id_vars=["species", "unit"], var_name="technology")
        for unit in emi_raw.unit.unique().tolist():
            if unit not in self.mp.units():
                self.mp.add_unit(unit)
        emi_raw = emi_raw[(emi_raw["value"] != 0) & (emi_raw["species"] != "CO2")]
        rel_base = emi_raw.rename(columns={"species": "relation"})

        rel_base.relation = rel_base.relation + "_Emission_bunkers"

        for relation_name in rel_base.relation.unique().tolist():
            if relation_name not in self.scen.set("relation").tolist():
                self.scen.add_set("relation", relation_name)

        rel = make_df("relation_activity", **rel_base, **common).round(7)
        rel = rel.pipe(broadcast, mode=modes)

        # exclude incompatible technology-mode permutations
        exclude = (rel["technology"].isin(["NH3_bunker", "LH2_bunker"])) & (
            (rel["mode"].str.contains("current"))
            | (rel["mode"].str.contains("oil"))
            | (rel["mode"].str.contains("solid"))
        )
        rel = rel[~exclude]
        rel = rel.pipe(broadcast, year_act=self.year_opt)
        rel["year_rel"] = rel["year_act"]

        self.scen.add_par("relation_activity", rel)

    def add_fuel_deployment_constraints(self):
        """
        Set up the scenario with growth activities.

        This function sets up various parameters for the scenario including growth
        activity/diffusion rate, activity constraints, introduction of fuels.
        """

        # Growth activity or diffusion rate
        base_df = {
            "node_loc": self.reg,
            "year_act": self.year_opt[1:],
            "time": "year",
            "unit": "%",
        }
        df_grow_lo = pd.DataFrame()
        df_grow_up = pd.DataFrame()
        for tec in fuel_list + occ_list:
            df = make_df(
                "growth_activity_lo",
                **base_df,
                technology=tec,
                value=self.input_parameters["transition_speed"]["min"][self.ssp],
            )
            df_grow_lo = pd.concat([df_grow_lo, df])
            df = make_df(
                "growth_activity_up",
                **base_df,
                technology=tec,
                value=self.input_parameters["transition_speed"]["max"][self.ssp],
            )
            df_grow_up = pd.concat([df_grow_up, df])

        self.scen.add_par("growth_activity_up", df_grow_up)
        self.scen.add_par("growth_activity_lo", df_grow_lo)

        # Activity constraint
        base_df = {
            "node_loc": self.reg,
            "year_act": self.year_opt,
            "time": "year",
            "unit": "GWa",
        }
        df_ini_up = pd.DataFrame()
        for tec in fuel_list + occ_list:
            df = make_df(
                "initial_activity_up",
                **base_df,
                technology=tec,
                value=self.input_parameters["initial_activity_up"]["max"][self.ssp],
            )
            df_ini_up = pd.concat([df_ini_up, df])
        self.scen.add_par("initial_activity_up", df_ini_up)

        # TODO: Add soft constraints

        # Introduction of fuels/ First year deployment
        # Set an upper bound of zero for the years
        df_bound_up = pd.DataFrame()
        ssp_constraint = self.input_parameters["introduction_fuels"][self.ssp]
        for fuel in ssp_constraint.keys():
            base_df = {
                "node_loc": self.reg,
                "year_act": ssp_constraint[fuel],
                "mode": "all",
                "time": "year",
                "unit": "GWa",
            }
            df = make_df(
                "bound_activity_up",
                **base_df,
                technology=fuel,
                value=np.zeros(len(ssp_constraint[fuel])),
            )
            df_bound_up = pd.concat([df_bound_up, df])
        self.scen.add_par("bound_activity_up", df_bound_up)

    def add_fuel_pathways(self):
        """
        Adds fuel pathways for new shipping types and modes.

        This method adds fuel pathways for different technologies,
        adjusting parameters based on specific conditions and sectors.
        """

        mode_params_dict = {}
        paramList_mode = [
            x for x in self.scen.par_list() if "mode" in self.scen.idx_sets(x)
        ]
        # paramList_mode = ["input", "output"]
        for fuel in ["foil", "LNG", "meth", "eth", "LH2", "NH3"]:  # "loil",
            for p in paramList_mode:
                if fuel == "NH3":
                    for sector, mode in zip(
                        self.ship_modes.keys(), self.ship_modes.values()
                    ):
                        if sector in ["shp_solid", "shp_oil"]:
                            pass
                        elif "current" in sector:
                            pass
                        elif self.ssp == "SSP3":
                            pass
                        else:
                            input = self.scen.par(
                                "input", filters={"technology": "foil_bunker"}
                            )
                            input_new = input.loc[input["mode"] == mode].copy()
                            input_new["technology"] = "NH3_bunker"
                            input_new["commodity"] = "NH3"
                            self.scen.add_par("input", input_new)

                            output = self.scen.par(
                                "output", filters={"technology": "foil_bunker"}
                            )
                            output = output.loc[output["commodity"] == sector]
                            output["technology"] = "NH3_bunker"
                            self.scen.add_par("output", output)
                else:
                    filter_param = self.scen.par(
                        p, filters={"technology": f"{fuel}_bunker"}
                    )
                    mode_params_dict[p] = filter_param
                    if len(filter_param) > 0:
                        for sector, mode in zip(
                            self.ship_modes.keys(), self.ship_modes.values()
                        ):
                            if (fuel == "LH2") and (sector in ["shp_solid", "shp_oil"]):
                                pass
                            elif (fuel == "LH2") and (self.ssp == "SSP3"):
                                pass
                            elif (fuel == "LH2") and ("current" in sector):
                                pass
                            else:
                                filter_param_new = filter_param.copy()
                                if (
                                    p == "historical_activity"
                                ):  # TODO Check if this is necessary at all
                                    pass
                                    # correct = dem_ship_fut[sector][0] / demand_year_2015
                                    # filter_param_new["value"] *= correct
                                    # filter_param_new["mode"] = mode
                                    # self.scen.add_par(p, filter_param_new)

                                    # filter_param_new["value"] *= 0
                                    # filter_param_new["mode"] = "M1"
                                    # self.scen.add_par(p, filter_param_new)
                                elif p == "ref_activity":
                                    pass
                                elif p == "relation_activity":
                                    pass
                                else:
                                    filter_param_new["mode"] = mode
                                    if "commodity" in filter_param.columns:
                                        if "shipping" in list(
                                            filter_param["commodity"]
                                        ):
                                            filter_param_new["commodity"] = sector
                                    self.scen.add_par(p, filter_param_new)
        # Remove M1 mode
        for fuel in fuel_list:
            for var in ["input", "output"]:
                remove = self.scen.par(var, filters={"technology": fuel, "mode": "M1"})
                self.scen.remove_par(var, remove)

    def baseline_2030(self):
        """
        Baseline 2030 with HFO+MGO (90%) and LNG (10%).

        It generates some strange results, so deactiviting this
        function at the momemtn.
        """

        # Defining share for all bunker fuels
        for shares, technology, share_val, type_tec_total, years_ac in zip(
            ["share_fossil_foil", "share_fossil_LNG", "share_LH2"],
            ["foil_bunker", "LNG_bunker", "LH2_bunker"],
            [1, 0.15, 0.1],
            [
                "maritime_fuel_total_foil",
                "maritime_fuel_total_LNG",
                "maritime_fuel_total_LH2",
            ],
            [
                [2035],  # [2020, 2025, 2030],
                [2035],  # [2020, 2025, 2030],
                [
                    2020,
                    2025,
                    2030,
                    2035,
                    2040,
                    2045,
                    2050,
                    2055,
                    2060,
                    2070,
                    2080,
                    2090,
                    2100,
                ],
            ],
        ):
            # Defining technologies that make up the total
            self.scen.add_cat("technology", type_tec_total, "foil_bunker")
            self.scen.add_cat("technology", type_tec_total, "loil_bunker")
            self.scen.add_cat("technology", type_tec_total, "LNG_bunker")
            self.scen.add_cat("technology", type_tec_total, "meth_bunker")
            self.scen.add_cat("technology", type_tec_total, "eth_bunker")
            self.scen.add_cat("technology", type_tec_total, "LH2_bunker")
            self.scen.add_cat("technology", type_tec_total, "NH3_bunker")
            self.scen.add_set("shares", shares)

            for ship_modes in self.ship_modes:
                if ("short" in ship_modes) and (technology == "LH2_bunker"):
                    pass
                else:
                    df = pd.DataFrame(
                        {
                            "shares": [shares],
                            "node_share": "R12_GLB",
                            "node": "R12_GLB",
                            "type_tec": type_tec_total,
                            "mode": ship_modes,
                            "commodity": ship_modes,
                            "level": "useful",
                        }
                    )
                    self.scen.add_set("map_shares_commodity_total", df)

                    # Defining technologies of share
                    type_tec = f"fossil_{technology}"
                    self.scen.add_cat("technology", type_tec, technology)

                    df = pd.DataFrame(
                        {
                            "shares": [shares],
                            "node_share": "R12_GLB",
                            "node": "R12_GLB",
                            "type_tec": type_tec,
                            "mode": ship_modes,
                            "commodity": ship_modes,
                            "level": "useful",
                        }
                    )
                    self.scen.add_set("map_shares_commodity_share", df)

                    # Defining the share
                    df = pd.DataFrame(
                        {
                            "shares": shares,
                            "node_share": "R12_GLB",
                            "year_act": years_ac,
                            "time": "year",
                            "value": [share_val] * len(years_ac),
                            "unit": "-",
                        }
                    )
                    # if technology != "LH2_bunker":
                    #    self.scen.add_par("share_commodity_lo", df)
                    self.scen.add_par("share_commodity_up", df)

    def connect_CO2_storage(self):
        """
        Configures maritime shipping technologies and CO2 balance relations in the scenario.
        Code adapted from a snippet provided by Yoga Pratama.

        """

        nodes = [node for node in self.scen.set("node") if node != "World"]
        years = [year for year in self.scen.set("year") if year > 2025]

        ## setup pipelines and storage technologies
        # filepath = "C:/Users/diogok/Downloads/GitHub/message_data_ntnu_shipping/message_data/projects/ntnu_shipping/data/ssp_data/data_default/maritime_setup_data_dev.yaml"
        filename = "maritime_setup_data_dev.yaml"
        # add_tech(self.scen, filepath=filepath)
        with open(self.data_path / filename, "r") as stream:
            tech_data = yaml.safe_load(stream)
        add_tech(self.scen, tech_data)

        # only allow activity when year_act == year_vtg
        reg_co2importer = ["marco2_import"]
        pars2remove = ["output"]
        for par in pars2remove:
            df = self.scen.par(par, {"technology": reg_co2importer})
            df = df.loc[df["year_vtg"] != df["year_act"]]
            self.scen.remove_par(par, df)

        # maritime technologies co2 emission
        # martechs_ems = {"mar_tech":-1}
        martechs_ems = {
            "foil_occ_bunker": -0.665
            * self.input_parameters["carbon_capture"][self.ssp]["cc_rate"]
            * self.input_parameters["carbon_capture"][self.ssp]["fuel_penalty"],
            "loil_occ_bunker": -0.631
            * self.input_parameters["carbon_capture"][self.ssp]["cc_rate"]
            * self.input_parameters["carbon_capture"][self.ssp]["fuel_penalty"],
            "LNG_occ_bunker": -0.482
            * self.input_parameters["carbon_capture"][self.ssp]["cc_rate"]
            * self.input_parameters["carbon_capture"][self.ssp]["fuel_penalty"],
        }

        for tec in list(martechs_ems.keys()):
            m1_mode = self.scen.par("bound_activity_up", filters={"technology": tec})
            self.scen.remove_par("bound_activity_up", m1_mode)
            for mode in self.ship_modes:
                m1_mode["mode"] = mode
                self.scen.add_par("bound_activity_up", m1_mode)

        # remove old setup
        # remove relations
        rels = [
            "co2_trans_disp",
            "bco2_trans_disp",
            "CO2_Emission_Global_Total",
            "CO2_Emission",
        ]
        self.scen.remove_par(
            "relation_activity",
            self.scen.par(
                "relation_activity",
                {"technology": list(martechs_ems.keys()), "relation": rels},
            ),
        )

        # setup CO2 balance between R12_GLB and other regions in maritime
        # adding set and dummy technologies
        if "marco2_import" not in self.scen.set("technology"):
            self.scen.add_set("technology", "marco2_import")
        if "mar_co2_balance" not in self.scen.set("relation"):
            self.scen.add_set("relation", "mar_co2_balance")

        # create relation activity
        list_relation = []
        for yr in years:
            for node in nodes:
                if node != "R12_GLB":
                    relact_co2stor = make_df(
                        "relation_activity",
                        relation="mar_co2_balance",
                        node_rel="R12_GLB",
                        year_rel=yr,
                        node_loc=node,
                        technology="marco2_import",
                        year_act=yr,
                        mode="M1",
                        value=1,
                        unit="-",
                    )
                    list_relation += [relact_co2stor]

            # for maritime technologies in R12_GLB
            for mode in self.ship_modes:
                for tech in martechs_ems.keys():
                    relact_co2stor_glb = make_df(
                        "relation_activity",
                        relation="mar_co2_balance",
                        node_rel="R12_GLB",
                        year_rel=yr,
                        node_loc="R12_GLB",
                        technology=tech,
                        year_act=yr,
                        mode=mode,
                        value=martechs_ems[tech],
                        unit="-",
                    )
                    list_relation += [relact_co2stor_glb]
                    relact_co2stor_glb = make_df(
                        "relation_activity",
                        relation="CO2_shipping",
                        node_rel="R12_GLB",
                        year_rel=yr,
                        node_loc="R12_GLB",
                        technology=tech,
                        year_act=yr,
                        mode=mode,
                        value=martechs_ems[tech],
                        unit="-",
                    )
                    list_relation += [relact_co2stor_glb]
        df_relation = pd.concat(list_relation)

        # create relation bounds
        df_rel_eq = make_df(
            "relation_upper",
            relation="mar_co2_balance",
            node_rel="R12_GLB",
            year_rel=years,
            value=0,
            unit="-",
        )

        # adding parameters
        self.scen.add_par("relation_activity", df_relation)
        self.scen.add_par("relation_upper", df_rel_eq)
        self.scen.add_par("relation_lower", df_rel_eq)

    def IMO_goal(self):
        """
        Set 2050 net-zero IMO goals.

        Set the emission targets and emission factors for the International
        Maritime Organization (IMO) goals in the scenario. Additionally, it
        adjusts for methane (CH4) and ammonia (N2O) emissions, converting
        them to their CO2 equivalent emissions, and sets the emission reduction
        goals for specific years as defined by the IMO net-zero targets.
        """

        # add all maritime fuels in the category "bunkers"
        self.scen.add_set("cat_tec", ["bunkers", "foil_tobunker"])
        self.scen.add_set("cat_tec", ["bunkers", "loil_tobunker"])
        self.scen.add_set("cat_tec", ["bunkers", "LNG_tobunker"])
        self.scen.add_set("cat_tec", ["bunkers", "foil_occ_bunker"])
        self.scen.add_set("cat_tec", ["bunkers", "loil_occ_bunker"])
        self.scen.add_set("cat_tec", ["bunkers", "LNG_occ_bunker"])
        self.scen.add_set("cat_tec", ["bunkers", "NH3_tobunker"])
        self.scen.add_set("cat_tec", ["bunkers", "eth_tobunker"])
        self.scen.add_set("cat_tec", ["bunkers", "meth_tobunker"])
        self.scen.add_set("cat_tec", ["bunkers", "LH2_tobunker"])

        # add emission, type and cat
        self.scen.add_set("type_emission", "CO2_shipping_IMO")
        self.scen.add_set("emission", "CO2_shipping_IMO")
        self.scen.add_set(
            "cat_emission",
            pd.DataFrame(
                data={
                    "type_emission": ["CO2_shipping_IMO"],
                    "emission": ["CO2_shipping_IMO"],
                }
            ),
        )

        # define base parameters
        base_emissions = {
            "node_loc": self.reg,
            "year_vtg": self.years,
            "year_act": self.years,
            #'unit': 'Mt CO2/GWa', # tC / kWyr ~ million ton C / GWa
        }
        # emission values shipping from:
        # https://docs.messageix.org/projects/global/en/stable/emissions/message/
        ch4_lng_gwp100 = 0.00645761454
        convert_n2o_co2eq = 0.35875636363
        emissions_shipping = {
            "foil_bunker": 0.665,
            "loil_bunker": 0.631,
            "LNG_bunker": 0.482,
            "foil_occ_bunker": -0.665
            * self.input_parameters["carbon_capture"][self.ssp]["cc_rate"],
            "loil_occ_bunker": -0.631
            * self.input_parameters["carbon_capture"][self.ssp]["cc_rate"],
            "LNG_occ_bunker": -0.482
            * self.input_parameters["carbon_capture"][self.ssp]["cc_rate"]
            + ch4_lng_gwp100,
            "eth_bunker": 0,
            "meth_bunker": 0,  # we assume that most methanol is biogenic
            "LH2_bunker": 0,
            "NH3_bunker": self.input_parameters["n2o_emissions"][self.ssp]
            * convert_n2o_co2eq,  # we assume most ammonia is blue/green with upstream CCS
        }
        for tec, val in emissions_shipping.items():
            output_fuel = self.scen.par("output", filters={"technology": tec})
            list_modes = list(output_fuel["mode"].unique())
            for mode in list_modes:
                df = make_df(
                    base_emissions,
                    technology=tec,
                    mode=mode,
                    emission="CO2_shipping_IMO",
                    value=val,
                )
                self.scen.add_par("emission_factor", df)

        # add net-zero goals with an error margin of 20MtC to
        # prevent unfeasibility
        years_imo_zero = [
            num
            for num in self.years
            if num >= self.input_parameters["year_netzero_emissions"][self.ssp]
        ]
        emission_goal = {
            "node": "R12_GLB",
            "type_emission": "CO2_shipping_IMO",
            "type_tec": "bunkers",
            "type_year": years_imo_zero,
            "unit": "MtC",
            "value": list(np.ones(len(years_imo_zero)) * 15),
        }
        self.scen.add_par("bound_emission", make_df(emission_goal))

    def remove_old_fuel_pathways(self):
        """
        Remove deprecated fuel pathways in M1.

        Removes outdated technology data from the scenario's parameters by
        filtering out entries from any maritime fuel still present in the
        "M1" mode.
        """

        # iterate through each parameter in the scenario's parameter list
        for par in self.scen.par_list():
            if "technology" in self.base.idx_names(
                par
            ) and "mode" in self.base.idx_names(par):
                df = self.scen.par(par, filters={"technology": fuel_list, "mode": "M1"})
                if not df.empty:
                    # filter out years before firstmodelyear
                    if "year_act" in self.base.idx_names(par):
                        df = df.loc[df.year_act >= self.base.firstmodelyear]
                    else:
                        df = df.loc[df.year_vtg > self.base.firstmodelyear]
                    if not df.empty:
                        # remove the fuels
                        self.scen.remove_par(par, df)
