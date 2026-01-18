/**
 * Overshoot Sentinel - Hazard Detection Module
 * Uses Overshoot SDK for real-time object detection
 */

class OvershootSentinel {
    constructor(config) {
        this.apiKey = config.apiKey;
        this.allergy = config.allergy || 'Monster energy drinks';
        this.medName = config.medName || 'Amoxicillin';
        this.medsTaken = config.medsTaken || false;
        this.onHazard = config.onHazard || (() => {});
        this.onLog = config.onLog || (() => {});
        this.onError = config.onError || console.error;
        this.debug = config.debug || false;
        
        this.vision = null;
        this.active = false;
    }

    buildPrompt() {
        const medStatus = this.medsTaken ? 'ALREADY TAKEN TODAY' : 'NOT YET TAKEN TODAY';
        
        return `You are a safety monitor. Analyze this image and respond with ONLY a JSON object.

PATIENT PROFILE:
- ALLERGIC TO: ${this.allergy}
- Medication: ${this.medName} (Status: ${medStatus})

LOOK FOR:
1. Monster Energy drink (green claw logo on black can) - this is a HAZARD
2. Any energy drink cans - HAZARD
3. Items related to "${this.allergy}" - HAZARD
4. Medication bottles when status is ALREADY TAKEN - HAZARD

RESPOND WITH ONLY ONE JSON (no other text):

If Monster Energy drink OR energy drink OR "${this.allergy}" item visible:
{"type": "hazard", "text": "Warning: ${this.allergy} detected!", "icon": "⚠️"}

If medication visible AND status is ALREADY TAKEN TODAY:
{"type": "hazard", "text": "Stop: Dose already taken today", "icon": "🚫"}

If medication visible AND status is NOT YET TAKEN TODAY:
{"type": "log", "text": "Medication logged successfully"}

If nothing hazardous detected:
{"type": "none"}`;
    }

    async start(videoElement) {
        // Check for SDK - it could be under different names
        const SDK = window.overshoot || window.Overshoot || window.RealtimeVision;
        const RealtimeVisionClass = SDK?.RealtimeVision || window.RealtimeVision;
        
        if (!RealtimeVisionClass) {
            console.error('Overshoot SDK not loaded. Available globals:', Object.keys(window).filter(k => k.toLowerCase().includes('over') || k.toLowerCase().includes('vision') || k.toLowerCase().includes('realtime')));
            this.onError(new Error('Overshoot SDK not loaded'));
            return false;
        }

        if (!this.apiKey) {
            console.error('Overshoot API key not provided');
            this.onError(new Error('API key required'));
            return false;
        }

        try {
            const prompt = this.buildPrompt();
            
            if (this.debug) {
                console.log('🛡️ Sentinel prompt:', prompt);
            }

            // Get the RealtimeVision class
            const SDK = window.overshoot || window.Overshoot || window;
            const RealtimeVisionClass = SDK.RealtimeVision || window.RealtimeVision;

            this.vision = new RealtimeVisionClass({
                apiUrl: 'https://api.overshoot.ai',
                apiKey: this.apiKey,
                prompt: prompt,
                source: {
                    type: 'camera',
                    cameraFacing: 'environment'
                },
                model: 'Qwen/Qwen3-VL-8B-Instruct',
                processing: {
                    fps: 2,
                    sampling_ratio: 0.5,
                    clip_length_seconds: 1.0,
                    delay_seconds: 1.0
                },
                debug: this.debug,
                onResult: (result) => this.handleResult(result),
                onError: (error) => {
                    console.error('🛡️ Sentinel error:', error);
                    this.onError(error);
                }
            });

            await this.vision.start();
            this.active = true;

            // Attach stream to video element for preview
            if (videoElement) {
                const stream = this.vision.getMediaStream();
                if (stream) {
                    videoElement.srcObject = stream;
                }
            }

            console.log('🛡️ Sentinel started successfully');
            return true;

        } catch (error) {
            console.error('🛡️ Failed to start Sentinel:', error);
            this.onError(error);
            return false;
        }
    }

    handleResult(result) {
        if (this.debug) {
            console.log('🛡️ Sentinel result:', result);
        }

        if (!result.ok) {
            console.warn('🛡️ Inference error:', result.error);
            return;
        }

        try {
            // Try to parse JSON from result
            let parsed;
            const text = result.result.trim();
            
            // Try direct JSON parse
            try {
                parsed = JSON.parse(text);
            } catch {
                // Try to extract JSON from response
                const jsonMatch = text.match(/\{[^}]+\}/);
                if (jsonMatch) {
                    parsed = JSON.parse(jsonMatch[0]);
                } else {
                    console.warn('🛡️ Could not parse response:', text);
                    return;
                }
            }

            // Handle the parsed result
            if (parsed.type === 'hazard') {
                console.log('🚨 HAZARD DETECTED:', parsed.text);
                this.onHazard(parsed);
            } else if (parsed.type === 'log') {
                console.log('✅ MEDICATION LOGGED:', parsed.text);
                this.onLog(parsed);
            } else if (this.debug && parsed.type === 'none') {
                console.log('🛡️ No hazard detected');
            }

        } catch (error) {
            console.warn('🛡️ Error processing result:', error, result.result);
        }
    }

    updateProfile(profile) {
        if (profile.allergy !== undefined) this.allergy = profile.allergy;
        if (profile.medName !== undefined) this.medName = profile.medName;
        if (profile.medsTaken !== undefined) this.medsTaken = profile.medsTaken;

        // Update prompt if vision is active
        if (this.vision && this.active) {
            const newPrompt = this.buildPrompt();
            this.vision.updatePrompt(newPrompt);
            console.log('🛡️ Profile updated');
        }
    }

    async stop() {
        if (this.vision) {
            try {
                await this.vision.stop();
                console.log('🛡️ Sentinel stopped');
            } catch (error) {
                console.warn('🛡️ Error stopping:', error);
            }
            this.vision = null;
        }
        this.active = false;
    }

    isActive() {
        return this.active;
    }
}

// Export for use
window.OvershootSentinel = OvershootSentinel;