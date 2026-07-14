import cv2
import math
import json
import os
from ultralytics import YOLO
from openai import OpenAI


client = OpenAI(api_key="  :)  ")

# This is our trained Yolo model(the bulk of the project)
model = YOLO('best.pt') 

# Video source 
VIDEO_SOURCE = "merlin_test_2.mp4" 

# How many frames to process before sending a report to the LLM
REPORTING_INTERVAL = 40  

# RUNWAY CONFIGURATION: well put the fictional runway in the center of the screen
RUNWAY_X = 390  
RUNWAY_Y = 340    

def generate_radar_report(telemetry_data):
    """Sends the compiled tracking data to OpenAI for a safety assessment."""
    if not telemetry_data:
        return "No targets currently tracked."

    data_str = json.dumps(telemetry_data, indent=2)
    
    messages = [
        {
            "role": "system", 
            "content": (
                "You are the Avian Radar Threat Assessment AI for a major commercial airport. Your primary directive is to eliminate 'Alert Fatigue' for Air Traffic Controllers (ATC).\n\n"
                "RULES OF ENGAGEMENT:\n"
                "1. SILENCE IS GOLDEN: Do NOT issue an alert if a target is moving away from the airport ('Outbound' heading) or moving too slow to intercept an aircraft.\n"
                "2. INTERSECTION IS KEY: A target is only a high threat if it is 'Inbound' to the runway and Speed > 30 knots.\n"
                "3. FLOCK MASS: Prioritize 'Large' size profiles.\n\n"
                "THREAT LEVELS:\n"
                "- [CLEAR]: No targets intersecting flight paths. Do not alert.\n"
                "- [ADVISORY]: Targets are in the airspace but not on a collision vector. (e.g., 'Flock of birds 2 miles North, Outbound. Monitor.')\n"
                "- [CRITICAL]: Target trajectory intersects active flight path. (e.g., 'ABORT TAKEOFF. Large flock Inbound, speed 40 knots.')\n\n"
                "OUTPUT FORMAT:\n"
                "Respond strictly in this format:\n"
                "[THREAT LEVEL] | [TARGET ID] | [ACTIONABLE ATC MESSAGE]"
            )
        },
        {"role": "user", "content": f"Current Radar Telemetry:\n{data_str}"}
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.2
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"API Error: {e}"
def main():
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    frame_count = 0
    target_telemetry = {}

    print("Starting Avian Radar Prototype...")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break # End of video
        
        frame_count += 1

        # Run YOLO tracking on the current frame
        results = model.track(frame, persist=True, tracker="botsort.yaml", verbose=False)
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box.tolist()
                
                # Calculate direct line-of-sight distance to the runway
                distance_to_runway = round(math.sqrt((x - RUNWAY_X)**2 + (y - RUNWAY_Y)**2), 1)
                
                if track_id in target_telemetry:
                    # Target exists: Calculate speed based on distance moved
                    old_x = target_telemetry[track_id]['x']
                    old_y = target_telemetry[track_id]['y']
                    old_distance = target_telemetry[track_id].get('Distance_To_Runway', distance_to_runway)
                    
                    pixel_distance = math.sqrt((x - old_x)**2 + (y - old_y)**2)
                    # Rough prototype conversion: 1 pixel moved per frame = ~2 knots, this could be fine-tuned to a specific avaian radar.
                    speed_knots = round(pixel_distance * 2.0, 1)
                    
                    # Determine Heading (Is it getting closer to the runway or further away...)
                    heading = "Inbound" if distance_to_runway < old_distance else "Outbound"
                    
                    target_telemetry[track_id].update({
                        "Status": "Tracking",
                        "Speed_Knots": speed_knots,
                        "Size_Profile": "Large" if w * h > 500 else "Small",
                        "Distance_To_Runway": distance_to_runway,
                        "Heading": heading,
                        "x": x, "y": y
                    })
                else:
                    # New target detected
                    target_telemetry[track_id] = {
                        "Status": "New Target", 
                        "Speed_Knots": 0, 
                        "Distance_To_Runway": distance_to_runway,
                        "Heading": "Unknown",
                        "x": x, "y": y
                    }

        # Visualize the tracking on screen
        annotated_frame = results[0].plot()
        
        # Draw a visual marker for the runway on the video feed(the big red dot on screen)
        cv2.circle(annotated_frame, (RUNWAY_X, RUNWAY_Y), 20, (0, 0, 255), -1)
        cv2.putText(annotated_frame, "RUNWAY", (RUNWAY_X - 45, RUNWAY_Y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Avian Radar Tracking", annotated_frame)

        # Trigger the LLM Report every N frames
        if frame_count % REPORTING_INTERVAL == 0:
            print(f"\n--- GENERATING REPORT AT FRAME {frame_count} ---")
            
            # Clean up the dictionary to only send relevant data (remove raw X/Y)
            clean_data = {}
            for tid, data in target_telemetry.items():
                if data["Status"] == "Tracking": # Only report relevent targets
                    clean_data[f"Target_{tid}"] = {
                        "Speed": data["Speed_Knots"],
                        "Size": data["Size_Profile"],
                        "Distance_To_Runway": data["Distance_To_Runway"],
                        "Heading": data["Heading"]
                    }
            
            report = generate_radar_report(clean_data)
            print(f"ATC ALERT: {report}")
            
            # Reset telemetry for the next interval to avoid outdated data
            target_telemetry.clear()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("System Shut Down.")

if __name__ == "__main__":
    main()