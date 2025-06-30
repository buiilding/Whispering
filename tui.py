import argparse
import sys
import os
import core
from que import Queue, Pair
import threading

# Suppress ALSA warnings by redirecting stderr
class SuppressALSA:
    def __enter__(self):
        self._stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stderr.close()
        sys.stderr = self._stderr

def main():
    parser = argparse.ArgumentParser(description="Transcribe and translate speech in real-time.")
    parser.add_argument('--mic', type=str, default=None, help='Microphone device name')
    parser.add_argument('--model', type=str, required=True, choices=core.MODELS, help='Model size to use')
    parser.add_argument('--device', type=str, default='auto', choices=core.DEVICES, help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--vad', action='store_true', help='Enable voice activity detection')
    parser.add_argument('--memory', type=int, default=3, help='Number of previous segments as prompt')
    parser.add_argument('--patience', type=float, default=5.0, help='Patience for segment completion (seconds)')
    parser.add_argument('--timeout', type=float, default=5.0, help='Timeout for translation service (seconds)')
    parser.add_argument('--prompt', type=str, default='', help='Initial prompt for transcription')
    parser.add_argument('--source', type=str, default=None, help='Source language (e.g., en)')
    parser.add_argument('--target', type=str, default=None, help='Target language (e.g., fr)')
    args = parser.parse_args()

    # Find mic index
    mic_index = None
    if args.mic is not None:
        try:
            mic_index = core.get_mic_index(args.mic)
        except Exception as e:
            print(f"[Error] Microphone '{args.mic}' not found: {e}", file=sys.stderr)
            sys.exit(1)

    tsres_queue = Queue[Pair]()
    tlres_queue = Queue[Pair]()
    stop_event = threading.Event()

    def on_success(proc):
        print("[INFO] Transcription started. Press Ctrl+C to stop.")
        def print_results():
            while not stop_event.is_set():
                if tsres_queue:
                    res = tsres_queue.get()
                    if res is None:
                        break
                    print(f"[Transcription] {res.done}{res.curr}")
                if args.target:
                    if tlres_queue:
                        res = tlres_queue.get()
                        if res is None:
                            break
                        print(f"[Translation]   {res.done}{res.curr}")
        threading.Thread(target=print_results, daemon=True).start()

    def on_failure(err):
        print(f"[Error] {err}", file=sys.stderr)
        stop_event.set()

    with SuppressALSA():
        core.start(
            index=mic_index,
            model=args.model,
            device=args.device,
            vad=args.vad,
            prompts=[args.prompt],
            memory=args.memory,
            patience=args.patience,
            timeout=args.timeout,
            source=args.source,
            target=args.target,
            tsres_queue=tsres_queue,
            tlres_queue=tlres_queue,
            on_success=on_success,
            on_failure=on_failure,
            log_cc_errors=lambda e: print(f"[Mic Error] {e}", file=sys.stderr),
            log_ts_errors=lambda e: print(f"[Transcription Error] {e}", file=sys.stderr),
            log_tl_errors=lambda e: print(f"[Translation Error] {e}", file=sys.stderr),
        )
        try:
            while True:
                if stop_event.is_set():
                    break
                threading.Event().wait(0.1)
        except KeyboardInterrupt:
            print("\n[INFO] Stopping...")
            stop_event.set()

if __name__ == "__main__":
    main() 