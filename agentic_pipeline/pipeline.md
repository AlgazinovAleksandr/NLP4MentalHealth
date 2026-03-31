Questionnaire answers
          │                                                                                                                                                        
          ▼
  LLM #1 — Interviewer agent                                                                                                                                       
    • Reads questionnaire results                                                                                                                                  
    • Asks targeted follow-up questions
      (driven by concern_areas + PHQ/GAD scores)                                                                                                                   
    • Conversation continues until enough signal collected                                                                                                       
          │                                                                                                                                                        
          ▼                                                                                                                                                      
    Full context:                                                                                                                                                  
    questionnaire answers + interview transcript                                                                                                                   
          │                                                                                                                                                        
          ▼                                                                                                                                                        
  LLM #2 — Diagnostic reasoning agent                                                                                                                              
    • Synthesizes all evidence                                                                                                                                     
    • Outputs probabilistic symptom overview                                                                                                                       
      e.g. { GAD: 0.82, MDD: 0.45, OCD: 0.21, healthy: 0.05 }                                                                                                      
    • Optionally: flags for human referral if uncertain or high-risk                                                                                               
                                                                                                                                                                   
  Two things worth deciding now before building:                                                                                                                   
                                                                                                                                                                   
  1. When does LLM #1 stop asking questions? You need a stopping condition — either a fixed number of turns, or LLM #1 itself decides it has enough signal         
  (self-assessed confidence). The second is more natural but harder to control.                                                                                  
  2. What format does LLM #2 output? A raw probability distribution is useful for the system, but the user-facing output should be something softer — e.g. "it     
  looks like anxiety and low mood are the main themes for you" — not a clinical diagnosis label. The probabilities drive the downstream agent behavior, not the    
  user display.