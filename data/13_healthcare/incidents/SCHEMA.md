# Incident Record Schema — IoMT Compliance Study

Each incident record is a structured narrative of one real or realistic security event
on an IoMT device or network. Records must be **human-authored** — this is stated in
the paper's methodology.

## Field definitions

| Field | Type | Description |
|---|---|---|
| `incident_id` | string | Unique ID, format `INC-NNN` |
| `attack_type` | string | One of the 18 CICIoMT2024 attack types |
| `attack_category` | string | DDoS / DoS / Recon / MQTT / Spoofing / BLE |
| `timestamp` | string | ISO 8601, e.g. `2024-03-15T09:42:00Z` |
| `affected_device` | string | IoMT device type (e.g. "Infusion pump", "Pulse oximeter") |
| `protocol` | string | Wi-Fi / MQTT / BLE / TCP/IP |
| `observed_behaviour` | string | What happened — the raw observable facts, 3–6 sentences |
| `actions_taken` | string | Response steps taken by staff or systems, 2–4 sentences |
| `outcome` | string | Final state — contained / data lost / device offline / etc. |
| `attck_technique` | string | MITRE ATT&CK for ICS technique ID + name |
| `controls_evidenced` | list | Which HIPAA/ISO atom IDs this record can evidence |
| `authored_by` | string | "human" always |

## Writing rules

1. **observed_behaviour** must be specific enough that an atom can be answered Yes/No.
   Bad:  "The device was attacked."
   Good: "At 09:42 UTC, the infusion pump's MQTT broker received 12,400 CONNECT packets
         per second from IP 10.0.2.44, causing the broker to exhaust its connection table
         and drop legitimate device telemetry for 4 minutes."

2. **actions_taken** must include at least one observable technical action
   (log entry, firewall rule, device restart, credential rotation, patch applied).

3. **outcome** must state whether ePHI was exposed, altered, or remained protected.
   This is what the atom "The covered entity shall implement technical security measures
   to guard against unauthorized access to ePHI being transmitted" is graded against.

4. Each record should naturally evidence 3–6 atoms. Do not try to cram all 76 atoms
   into one record.

5. Vary device types across records: infusion pump, pulse oximeter, ECG monitor,
   insulin pump, ventilator, MRI scanner gateway, patient wearable, MQTT broker,
   BLE-enabled pill dispenser.

## Target: 50 records total
- 2–3 records per attack type
- Mix: ~60% attack contained (controls evidenced as addressed),
       ~40% attack succeeded (controls evidenced as not addressed / partially addressed)
  This ratio replicates the NA-bias setting and gives the model both positive and
  negative examples to grade against.
