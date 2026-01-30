"""
ByteTrack implementation for person tracking.

Reused from vehicle tracking with configuration tuned for people
(slower movement, more jitter-tolerant).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple


@dataclass
class STrack:
    """Single object tracking representation for ByteTrack"""
    track_id: int
    bbox: List[float]  # [x, y, w, h]
    score: float
    class_id: int
    class_name: str
    state: str = 'tracked'  # 'tracked', 'lost', 'removed'
    frame_id: int = 0
    start_frame: int = 0
    tracklet_len: int = 0
    
    # Kalman filter state (simplified - just velocity)
    velocity: List[float] = field(default_factory=lambda: [0, 0, 0, 0])
    
    def predict(self):
        """Predict next position based on velocity"""
        self.bbox = [
            self.bbox[0] + self.velocity[0],
            self.bbox[1] + self.velocity[1],
            self.bbox[2] + self.velocity[2],
            self.bbox[3] + self.velocity[3]
        ]
    
    def update(self, new_bbox: List[float], new_score: float, frame_id: int):
        """Update track with new detection"""
        # Update velocity (simple moving average)
        alpha = 0.3
        self.velocity = [
            alpha * (new_bbox[i] - self.bbox[i]) + (1 - alpha) * self.velocity[i]
            for i in range(4)
        ]
        self.bbox = new_bbox
        self.score = new_score
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.state = 'tracked'


class ByteTracker:
    """
    ByteTrack: Multi-Object Tracking by Associating Every Detection Box
    
    Key idea: Use both high and low confidence detections for tracking.
    - First associate high confidence detections with existing tracks
    - Then associate low confidence detections with remaining unmatched tracks
    - This helps maintain tracks during occlusion
    """
    
    def __init__(self, 
                 track_thresh: float = 0.5,      # High confidence threshold
                 track_buffer: int = 30,          # Frames to keep lost tracks
                 match_thresh: float = 0.8,       # IoU threshold for matching
                 low_thresh: float = 0.1):        # Low confidence threshold
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.low_thresh = low_thresh
        
        self.tracked_stracks: List[STrack] = []
        self.lost_stracks: List[STrack] = []
        self.removed_stracks: List[STrack] = []
        
        self.frame_id = 0
        self.next_id = 1
    
    def _get_next_id(self) -> int:
        ret = self.next_id
        self.next_id += 1
        return ret
    
    @staticmethod
    def iou_distance(tracks: List[STrack], detections: List[dict]) -> np.ndarray:
        """Calculate IoU distance matrix between tracks and detections"""
        if len(tracks) == 0 or len(detections) == 0:
            return np.zeros((len(tracks), len(detections)))
        
        cost_matrix = np.zeros((len(tracks), len(detections)))
        
        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                # Calculate IoU
                t_box = track.bbox  # [x, y, w, h]
                d_box = det['bbox']  # [x, y, w, h]
                
                # Convert to [x1, y1, x2, y2]
                t_x1, t_y1 = t_box[0], t_box[1]
                t_x2, t_y2 = t_box[0] + t_box[2], t_box[1] + t_box[3]
                d_x1, d_y1 = d_box[0], d_box[1]
                d_x2, d_y2 = d_box[0] + d_box[2], d_box[1] + d_box[3]
                
                # Intersection
                inter_x1 = max(t_x1, d_x1)
                inter_y1 = max(t_y1, d_y1)
                inter_x2 = min(t_x2, d_x2)
                inter_y2 = min(t_y2, d_y2)
                
                if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
                    iou = 0.0
                else:
                    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                    t_area = t_box[2] * t_box[3]
                    d_area = d_box[2] * d_box[3]
                    union_area = t_area + d_area - inter_area
                    iou = inter_area / union_area if union_area > 0 else 0.0
                
                cost_matrix[i, j] = 1 - iou  # Cost = 1 - IoU
        
        return cost_matrix
    
    @staticmethod
    def linear_assignment(cost_matrix: np.ndarray, thresh: float) -> Tuple[List, List, List]:
        """Simple greedy assignment"""
        if cost_matrix.size == 0:
            return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))
        
        matches = []
        unmatched_tracks = list(range(cost_matrix.shape[0]))
        unmatched_dets = list(range(cost_matrix.shape[1]))
        
        # Greedy matching
        while True:
            if len(unmatched_tracks) == 0 or len(unmatched_dets) == 0:
                break
            
            # Find minimum cost
            min_cost = float('inf')
            min_i, min_j = -1, -1
            
            for i in unmatched_tracks:
                for j in unmatched_dets:
                    if cost_matrix[i, j] < min_cost:
                        min_cost = cost_matrix[i, j]
                        min_i, min_j = i, j
            
            if min_cost > thresh or min_i == -1:
                break
            
            matches.append((min_i, min_j))
            unmatched_tracks.remove(min_i)
            unmatched_dets.remove(min_j)
        
        return matches, unmatched_tracks, unmatched_dets
    
    def update(self, detections: List[dict]) -> Dict[int, dict]:
        """
        Update tracker with new detections
        
        Args:
            detections: List of detection dicts with 'bbox' [x, y, w, h], 'confidence', 'class_id', 'class_name'
        
        Returns:
            Dictionary of active tracks {track_id: track_info}
        """
        self.frame_id += 1
        
        # Separate high and low confidence detections
        high_dets = [d for d in detections if d['confidence'] >= self.track_thresh]
        low_dets = [d for d in detections if self.low_thresh <= d['confidence'] < self.track_thresh]
        
        # Predict new locations for existing tracks
        for track in self.tracked_stracks:
            track.predict()
        
        # ----- First association: high confidence detections with tracked tracks -----
        cost_matrix = self.iou_distance(self.tracked_stracks, high_dets)
        matches, u_tracks, u_dets = self.linear_assignment(cost_matrix, 1 - self.match_thresh)
        
        # Update matched tracks
        for track_idx, det_idx in matches:
            track = self.tracked_stracks[track_idx]
            det = high_dets[det_idx]
            track.update(det['bbox'], det['confidence'], self.frame_id)
        
        # ----- Second association: low confidence detections with unmatched tracks -----
        remaining_tracks = [self.tracked_stracks[i] for i in u_tracks]
        cost_matrix = self.iou_distance(remaining_tracks, low_dets)
        matches2, u_tracks2, _ = self.linear_assignment(cost_matrix, 1 - 0.5)  # Lower IoU threshold
        
        for track_idx, det_idx in matches2:
            track = remaining_tracks[track_idx]
            det = low_dets[det_idx]
            track.update(det['bbox'], det['confidence'], self.frame_id)
            u_tracks.remove(self.tracked_stracks.index(track))
        
        # ----- Handle unmatched tracks -----
        for track_idx in u_tracks:
            track = self.tracked_stracks[track_idx]
            track.state = 'lost'
        
        # Move lost tracks
        new_lost = [self.tracked_stracks[i] for i in u_tracks]
        self.lost_stracks.extend(new_lost)
        
        # ----- Third association: unmatched high dets with lost tracks -----
        remaining_dets = [high_dets[i] for i in u_dets]
        cost_matrix = self.iou_distance(self.lost_stracks, remaining_dets)
        matches3, u_lost, u_dets3 = self.linear_assignment(cost_matrix, 1 - self.match_thresh)
        
        reactivated = []
        for track_idx, det_idx in matches3:
            track = self.lost_stracks[track_idx]
            det = remaining_dets[det_idx]
            track.update(det['bbox'], det['confidence'], self.frame_id)
            reactivated.append(track)
        
        # ----- Create new tracks for remaining unmatched detections -----
        new_tracks = []
        for det_idx in u_dets3:
            det = remaining_dets[det_idx]
            new_track = STrack(
                track_id=self._get_next_id(),
                bbox=det['bbox'],
                score=det['confidence'],
                class_id=det['class_id'],
                class_name=det['class_name'],
                frame_id=self.frame_id,
                start_frame=self.frame_id,
                tracklet_len=1
            )
            new_tracks.append(new_track)
        
        # ----- Update track lists -----
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == 'tracked']
        self.tracked_stracks.extend(reactivated)
        self.tracked_stracks.extend(new_tracks)
        
        # Remove old lost tracks
        self.lost_stracks = [t for t in self.lost_stracks 
                           if t not in reactivated and 
                           self.frame_id - t.frame_id < self.track_buffer]
        
        # Return active tracks as dict
        result = {}
        for track in self.tracked_stracks:
            result[track.track_id] = {
                'bbox': track.bbox,  # [x, y, w, h]
                'class_name': track.class_name,
                'confidence': track.score,
                'age': track.tracklet_len,
                'class_id': track.class_id
            }
        
        return result


def create_person_tracker():
    """
    Create a ByteTracker instance configured for person tracking.
    
    People move slower and have more jitter than vehicles, so we use:
    - Lower match_thresh (0.6 instead of 0.8): More lenient matching for jitter
    - Longer track_buffer (50 instead of 30): Keep lost tracks longer
    - Lower track_thresh (0.4 instead of 0.5): Accept lower confidence detections
    """
    return ByteTracker(
        track_thresh=0.4,      # Lower threshold for people
        track_buffer=50,        # Keep lost tracks longer (people move slower)
        match_thresh=0.6,       # More lenient matching (handle jitter better)
        low_thresh=0.1          # Low confidence threshold
    )

