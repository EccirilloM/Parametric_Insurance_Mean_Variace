import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from utils.plotter import run_simulation_and_plot, run_bonus_fat_tail_plot

def main():
    print("Starting simulation for Question 1...")

    path_case_A = os.path.join(BASE_DIR, "figurePlot03.png")
    path_case_B = os.path.join(BASE_DIR, "figurePlot0302.png")
    path_bonus = os.path.join(BASE_DIR, "figurePlotBonus.png")

    # Case A: theta_d = 30%, theta_p = 30%
    run_simulation_and_plot(
        theta_d=0.30, 
        theta_p=0.30, 
        output_path=path_case_A
    )

    
    # Case B: theta_d = 30%, theta_p = 20%
    run_simulation_and_plot(
        theta_d=0.30, 
        theta_p=0.20, 
        output_path=path_case_B
    )

    # Extra: Fat Tails Impact
    run_bonus_fat_tail_plot(
        theta_d=0.30,
        theta_p=0.30,
        output_path=path_bonus
    )

if __name__ == "__main__":
    main()