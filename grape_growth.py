def grape_growth_suggestion(temp, humidity, soil_ph, nitrogen):
    """
    Suggest if grapes can grow under given environmental conditions
    with detailed agronomic explanations.
    """
    messages = []

    if 18 <= temp <= 30:
        messages.append(
            "Temperature is ideal.\n"
            "  • Effect: Supports healthy photosynthesis and enzyme activity.\n"
            "  • Impact: Promotes balanced vine growth and fruit development."
        )
    elif temp < 18:
        messages.append(
            "Temperature is too low.\n"
            "  • Effect: Slows metabolic and enzymatic activity.\n"
            "  • Impact: Delayed growth, poor flowering, and reduced yield."
        )
    else:
        messages.append(
            "Temperature is too high.\n"
            "  • Effect: Causes heat stress and excessive transpiration.\n"
            "  • Impact: Leaf scorch, poor berry set, and reduced fruit quality."
        )

    if 40 <= humidity <= 70:
        messages.append(
            "Humidity is optimal.\n"
            "  • Effect: Maintains balanced transpiration.\n"
            "  • Impact: Reduces stress while limiting disease development."
        )
    elif humidity < 40:
        messages.append(
            "Humidity is too low.\n"
            "  • Effect: Increases water loss through transpiration.\n"
            "  • Impact: Vine stress, leaf curling, and reduced photosynthesis."
        )
    else:
        messages.append(
            "Humidity is too high.\n"
            "  • Effect: Creates favorable conditions for fungal pathogens.\n"
            "  • Impact: Increased risk of powdery mildew, downy mildew, and gray mold."
        )

    if 5.5 <= soil_ph <= 7:
        messages.append(
            "Soil pH is suitable.\n"
            "  • Effect: Ensures optimal nutrient availability.\n"
            "  • Impact: Healthy root development and efficient nutrient uptake."
        )
    elif soil_ph < 5.5:
        messages.append(
            "Soil is too acidic.\n"
            "  • Effect: Nutrient lockout, especially calcium and magnesium.\n"
            "  • Impact: Weak vine growth and poor fruit development."
        )
    else:
        messages.append(
            "Soil is too alkaline.\n"
            "  • Effect: Reduced availability of iron and micronutrients.\n"
            "  • Impact: Leaf chlorosis and reduced photosynthetic efficiency."
        )

    if 20 <= nitrogen <= 50:
        messages.append(
            "Nitrogen levels are optimal.\n"
            "  • Effect: Supports balanced vegetative and reproductive growth.\n"
            "  • Impact: Good canopy structure and high-quality fruit production."
        )
    elif nitrogen < 20:
        messages.append(
            "Nitrogen is low.\n"
            "  • Effect: Limits protein synthesis and leaf growth.\n"
            "  • Impact: Poor vine vigor, yellowing leaves, and low yield."
        )
    else:
        messages.append(
            "Nitrogen is high.\n"
            "  • Effect: Excessive vegetative growth (more leaves).\n"
            "  • Impact: Poor fruit quality, delayed ripening, and disease susceptibility."
        )

    if all([
        18 <= temp <= 30,
        40 <= humidity <= 70,
        5.5 <= soil_ph <= 7,
        20 <= nitrogen <= 50
    ]):
        messages.append(
            "Overall Assessment: Conditions are suitable for grape growth.\n"
            "  • Recommendation: Maintain current environmental and soil conditions."
        )
    else:
        messages.append(
            "Overall Assessment: Conditions are not ideal for grape growth.\n"
            "  • Recommendation: Adjust the above factors to improve vine health and yield."
        )

    return "\n\n".join(messages)


if __name__ == "__main__":
    temp = float(input("Enter Temperature (°C): "))
    humidity = float(input("Enter Humidity (%): "))
    soil_ph = float(input("Enter Soil pH: "))
    nitrogen = float(input("Enter Nitrogen level (ppm): "))

    suggestion = grape_growth_suggestion(temp, humidity, soil_ph, nitrogen)
    print("\n" + suggestion)



def pest_risk_assessment(
    disease_name,
    temp,
    humidity,
    soil_ph,
    nitrogen
):
    """
    Rule-based pest risk prediction and control advice.
    """

    risk_score = 0
    reasons = []
    control = []

    if humidity > 70:
        risk_score += 2
        reasons.append("High humidity favors insect breeding and fungal-associated pests.")

    if temp > 28:
        risk_score += 1
        reasons.append("Warm temperatures accelerate pest life cycles.")

    if nitrogen > 80:
        risk_score += 1
        reasons.append("Excess nitrogen promotes soft tissue growth attractive to pests.")

    if soil_ph < 5.5 or soil_ph > 7.5:
        risk_score += 1
        reasons.append("Soil pH imbalance weakens plant immunity.")

    if disease_name.lower() != "healthy":
        risk_score += 2
        reasons.append(
            "Existing disease weakens plant defense, increasing susceptibility to pests."
        )
        control.append(
            "Remove and destroy infected plant parts to reduce pest shelter."
        )

    if risk_score <= 1:
        risk = "Low"
    elif risk_score <= 3:
        risk = "Moderate"
    else:
        risk = "High"

    if risk in ["Moderate", "High"]:
        control.extend([
            "Use neem oil or insecticidal soap as a preventive spray.",
            "Introduce biological controls like lady beetles or lacewings.",
            "Avoid overhead irrigation to reduce pest-friendly humidity.",
            "Monitor undersides of leaves regularly for early infestation."
        ])

    if risk == "High":
        control.append(
            "If infestation is severe, apply recommended chemical pesticides "
            "following agricultural safety guidelines."
        )

    message = (
        f"Pest Risk Assessment:\n"
        f"Risk Level: {risk}\n\n"
        f"Reasons:\n- " + "\n- ".join(reasons) + "\n\n"
        f"Pest Management Recommendations:\n- " + "\n- ".join(control)
    )

    return message
