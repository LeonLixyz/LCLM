"""Submit one full-generation job to the deployed Modal app, then exit.

Run the checkpoint audit and inspect current jobs BEFORE invoking this command.
This is an explicit submission, not an idempotent 'ensure running' operation.
Keep the returned call ID for status checks; never resubmit on a polling timeout.
"""
import json

import modal

from data.generate_real_expansion_modal import APP_NAME


def main():
    generator = modal.Cls.from_name(APP_NAME, "RealExpansionGenerator")()
    call = generator.generate_full.spawn()
    print(json.dumps({"app": APP_NAME, "call_id": call.object_id,
                      "status": "submitted", "waited_for_completion": False}), flush=True)


if __name__ == "__main__":
    main()
