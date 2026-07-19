import subprocess

#Loop 10 time
for j in range(0, 10): 
    for i in range(0, 1): # This outer loop (for me to do experiment)
        
        # if i == 1:
        #     continue

        
        # Build the command exactly as you would type it in the terminal
        # Change the hide and seek here
        command = [
            "python", "arena.py", 
            "--seek", "bfs-son",
            "--hide", "24127552"
        ]
        
        print(f"Running iteration {j}...")

        # Run the command and wait for it to finish before the next loop
        subprocess.run(command)