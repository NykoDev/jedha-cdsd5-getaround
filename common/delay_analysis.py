import pandas as pd


def load_delay_data(path="data/get_around_delay_analysis.xlsx"):
    """Charge le dataset des retards et dérive is_late (booléen nullable, NA non imputé)."""
    dfda = pd.read_excel(path)

    dfda["is_late"] = (dfda["delay_at_checkout_in_minutes"] > 0).astype("boolean")
    dfda.loc[dfda["delay_at_checkout_in_minutes"].isna(), "is_late"] = pd.NA

    return dfda


def build_chained(dfda):
    """Isole les locations chaînées à une précédente location du même véhicule (<12h)
    et calcule checkin_impacted (le checkin suivant a-t-il été réellement retardé ?)."""
    chained = dfda[dfda["previous_ended_rental_id"].notna()].copy()

    delay_by_rental = dfda.set_index("rental_id")["delay_at_checkout_in_minutes"]
    chained["previous_delay"] = chained["previous_ended_rental_id"].astype(int).map(delay_by_rental)

    chained["checkin_impacted"] = (
        chained["previous_delay"] > chained["time_delta_with_previous_rental_in_minutes"]
    ).astype("boolean")
    chained.loc[chained["previous_delay"].isna(), "checkin_impacted"] = pd.NA

    return chained


def load_price_proxy(path="data/get_around_pricing_project.csv"):
    """Médiane du prix/jour, utilisée comme proxy de revenu (pas de clé de jointure entre les 2 datasets)."""
    dfp = pd.read_csv(path, index_col=0)
    return dfp["rental_price_per_day"].median()


def simulate(dfda, chained, median_price_per_day, threshold, scope):
    """
    Simule l'impact d'un seuil de tolérance sur les locations chaînées.
    threshold : seuil de tolérance en minutes
    scope : 'connect_only' (locations connect uniquement), 'mobile_only' (locations mobile
    uniquement), 'all' (toutes les locations)
    """
    if scope == "connect_only":
        scope_mask_full = dfda["checkin_type"] == "connect"
        scope_mask_chained = chained["checkin_type"] == "connect"
    elif scope == "mobile_only":
        scope_mask_full = dfda["checkin_type"] == "mobile"
        scope_mask_chained = chained["checkin_type"] == "mobile"
    else:
        scope_mask_full = pd.Series(True, index=dfda.index)
        scope_mask_chained = pd.Series(True, index=chained.index)

    total_rentals = scope_mask_full.sum()
    total_ended_revenue = (scope_mask_full & (dfda["state"] == "ended")).sum() * median_price_per_day

    affected_mask = scope_mask_chained & (chained["time_delta_with_previous_rental_in_minutes"] < threshold)
    n_affected = affected_mask.sum()
    pct_affected = n_affected / total_rentals * 100

    revenue_at_risk = n_affected * median_price_per_day
    pct_revenue_at_risk = revenue_at_risk / total_ended_revenue * 100

    problematic_mask = scope_mask_chained & (chained["checkin_impacted"] == True)
    n_problematic = problematic_mask.sum()
    n_resolved = (problematic_mask & (chained["time_delta_with_previous_rental_in_minutes"] < threshold)).sum()
    n_remaining = n_problematic - n_resolved
    pct_problematic_resolved = n_resolved / n_problematic * 100

    return {
        "threshold": threshold,
        "scope": scope,
        "total_rentals": total_rentals,
        "n_affected": n_affected,
        "pct_affected": pct_affected,
        "revenue_at_risk": revenue_at_risk,
        "pct_revenue_at_risk": pct_revenue_at_risk,
        "n_problematic": n_problematic,
        "n_resolved": n_resolved,
        "n_remaining": n_remaining,
        "pct_problematic_resolved": pct_problematic_resolved,
    }


def run_simulation_grid(dfda, chained, median_price_per_day, thresholds=None, scopes=("all", "connect_only", "mobile_only")):
    """Calcule simulate() sur une grille de seuils x scopes, retourne le DataFrame results."""
    if thresholds is None:
        max_time_delta = int(chained["time_delta_with_previous_rental_in_minutes"].max())
        thresholds = list(range(0, max_time_delta + 30, 30))

    return pd.DataFrame(
        [simulate(dfda, chained, median_price_per_day, t, s) for t in thresholds for s in scopes]
    )
