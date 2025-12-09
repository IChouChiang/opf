# Testing Guide: PYPOWER Visualization Tool

## How to Test N-1 Contingency Features

### 1. **Basic Network Viewing (Already Working!)**
   - You should see 39 buses displayed in different shapes:
     - **Blue circles** = PQ buses (load)
     - **Green diamonds** = PV buses (generator)  
     - **Red star** = Slack bus (reference, bus 39)
   - Buses with yellow dots have generators attached
   - All 46 branches should be visible as gray lines

### 2. **Interactive Graph Controls**
   
   **Test Dragging:**
   - Click and hold any bus node, drag it to a new position
   - The force simulation will adjust the network layout
   - Release to let it settle into place
   
   **Test Zooming:**
   - Scroll your mouse wheel over the graph
   - You should zoom in/out smoothly
   
   **Test Panning:**
   - Click and drag the graph background (not on a node)
   - The entire view shifts
   
   **Test Tooltips:**
   - Hover your mouse over any bus
   - A popup should show: Bus ID, Type, Load (MW), Voltage (p.u.)
   - If disconnected, Island number also shows

### 3. **Single Branch Outage (Simple Test)**
   
   **Step-by-step:**
   1. Look at the **Branch Control** panel (left side)
   2. You should see "Active: 46 / 46 branches" at the top
   3. Scroll through the branch list
   4. Click on any branch (e.g., "Branch 0: Bus 1 → Bus 2")
   5. **What to expect:**
      - The toggle switch turns gray
      - That branch disappears from the visualization
      - "Active" count drops to "45 / 46 branches"
      - **Connectivity Indicator** should still show "Connected" (green) because case39 is well-connected
   6. Click the branch again to restore it

### 4. **Multiple Branch Outages (Network Split Test)**
   
   To intentionally split the network and see islands:
   
   **Method 1: Hide Many Branches**
   1. Click the **"Hide All"** button in Branch Control
   2. **What to expect:**
      - All branches disappear from graph
      - Buses become isolated points
      - **Connectivity status changes to "Disconnected" (red)**
      - Shows "Islands: 39" (each bus is its own island)
      - Yellow warning appears: "Network is split into multiple islands"
      - Each bus gets a different color (island coloring)
   3. Click **"Show All"** to restore
   
   **Method 2: Selective Outages** (More realistic)
   1. Use the search box to find branches connected to a specific bus
      - Type "1" to search for Bus 1
      - You'll see branches like "Bus 1 → Bus 2", "Bus 1 → Bus 39"
   2. Hide all branches connected to that bus
   3. **What to expect:**
      - That bus becomes isolated (different color)
      - Connectivity shows "2 islands"
      - Main network + isolated bus
      - Island details show: "Island 1: 38 buses, Island 2: 1 bus"

### 5. **Search and Filter Features**
   
   **Test Search:**
   - Type a bus number in the search box (e.g., "30")
   - Only branches involving Bus 30 appear
   - Clear search to see all branches again
   
   **Test Active Filter:**
   - Hide several branches
   - Click the **"Active Only"** button
   - Only active (visible) branches show in the list
   - Click again to show all branches (including hidden ones)

### 6. **Real-Time Connectivity Analysis**
   
   **Watch it update:**
   1. Start with all branches active (connected network)
   2. Slowly hide branches one by one
   3. Watch the **Connectivity Indicator** in real-time:
      - It runs BFS algorithm after every change
      - Status updates immediately
      - Island count updates
      - Graph colors change when network splits

### 7. **System Info Panel**
   
   Check the right panel shows:
   - Base MVA: 100
   - Bus Types breakdown (29 PQ, 9 PV, 1 Slack)
   - Total Load: ~6,254.23 MW
   - Gen Capacity: ~8,200 MW

## Expected Behavior Summary

| Action | Expected Result |
|--------|----------------|
| Load page | 39 buses, 46 branches, Connected status |
| Hide 1 branch | Still connected (case39 is robust) |
| Hide All | 39 islands, red "Disconnected" warning |
| Show All | Back to 1 island, green "Connected" |
| Drag node | Node moves, links follow |
| Zoom | Graph scales smoothly |
| Hover bus | Tooltip appears with bus info |
| Search "30" | Filters to branches with Bus 30 |

## Common Test Scenarios

### Scenario A: Find Critical Links
- Hide branches one at a time
- Most single outages won't split the network
- Case39 is designed to be resilient

### Scenario B: Radial Outage
- Find a "leaf" bus (bus with only one connection)
- Hide its only branch
- That bus becomes isolated island

### Scenario C: Major Disturbance
- Hide 5-10 branches simultaneously
- See if network fragments
- Observe island coloring and sizes

## Tips for Better Testing

1. **Use browser at full screen** for best view
2. **Zoom in** to see bus labels clearly
3. **Drag nodes apart** if they overlap
4. **Use search** for specific buses
5. **Check island colors** - each island gets unique color when disconnected

## If Something Doesn't Work

- **No graph shown**: Check browser console (F12) for errors
- **Branches won't toggle**: Make sure you're clicking the branch item, not just the toggle
- **Laggy**: Try hiding some branches to reduce nodes displayed
- **Can't read text**: The UI is now more compact for 4K displays

Enjoy testing! 🚀
