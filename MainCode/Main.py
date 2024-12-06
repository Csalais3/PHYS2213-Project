from Section1 import VoiceRecorder, FourierDecomposition, FourierRecomposition
print("start")
from Section2 import RealtimeFT
from Section3 import RealtimeSinus
import time


def main():
    print("Welcome! This is my real-time voice decompisiton using Fast Fourier decomposition project for Intro to Modern Physics")
    print("Please select which section you would like to explore? (Section 1, Section 2, Section 3)")

    # User inputs section
    section = input().capitalize().replace(" ", "")
    if section.isnumeric():
        section = "Section" + section
    print(f"Thank you! Launching {section[:7] + " " + section[7:]}...")
    
    # Launches desired section
    if section == "Section1":
        VoiceRecorder.execute()
        FourierDecomposition.execute()
        FourierRecomposition.execute()
    elif section == "Section2":
        RealtimeFT.execute()
    elif section == "Section3":
        RealtimeSinus.execute()
    else:
        print(f"{section[:7] + " " + section[7:]} does not exist, please try another section.")

if __name__ == "__main__":
    main()