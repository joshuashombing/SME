import tkinter as tk


def open_inspection_window():
    # Function to open inspection window and close the main window
    main_window.destroy()

    from MultipleCamerasDemo import GUI, GUIAIEngine

    ai_engines = GUIAIEngine(number_of_camera=2)
    ai_engines.start_ai_engines()

    gui = GUI(ai_engines=ai_engines)
    gui.window.mainloop()


def open_train_window():
    # Function to open train window and close the main window
    main_window.destroy()

    from TrainModelsMultipleCamerasDemo import GUI, GUIAIEngine

    ai_engines = GUIAIEngine(number_of_camera=2)
    ai_engines.start_ai_engines()

    gui = GUI(ai_engines=ai_engines)
    gui.window.mainloop()


if __name__ == "__main__":
    # Creating the main window
    main_window = tk.Tk()
    main_window.title('SME Spring Sheet Metal Inspection')
    main_window.iconbitmap("satnusa.ico")
    window_w = 100
    window_h = 400
    main_window.geometry(f'{window_h}x{window_w}')
    main_window.resizable(False, False)  # Fixed sized

    # Creating a button to choose option one
    button_one = tk.Button(main_window, text="Inspection Mode", command=open_inspection_window)
    button_one.pack(pady=10)

    # Creating a button to choose option two
    button_two = tk.Button(main_window, text="Train Models Mode", command=open_train_window)
    button_two.pack(pady=10)

    main_window.mainloop()
