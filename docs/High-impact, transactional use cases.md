### High-impact, transactional use cases (agent executes/advances a business process)

1. **Auto Address/Contact Update Request via Agent -** For orders that have not yet been dispatched, the AI should be able to initiate an address change or provide a clear path to self-service modification. **Example Query:** "I need to change my delivery address before the order is shipped. How do I do that?"
2. **Invoice/GST Bill Request -** The AI should be able to instantly locate and email the invoice/warranty details based on the customer's order ID or registered email. **Example Query:** "Please send me the invoice/order details for my recent purchase of the MacBook."
3. **Live Stock Information/Product availability**
4. **Warranty claim intake & triage (with media capture)**
    - **Trigger:** “My mic isn’t working.”
    - **Flow:** Collect order#/phone#/IMEI → capture issue description → prompt for short issue video → package and email to care@grest.in with metadata → confirm ticket ID → surface next steps (pickup/repair guidance).
    - **Grounding:** Warranty steps explicitly require sharing a video + identifiers; functional issues covered.
    
    [Warranty Policy](https://grest.in/pages/warranty)
    

**4. Return initiation (RMA) within 7 days**

- **Trigger:** “I want to return this.”
- **Flow:** Validate window (delivery date vs today) → checklist for “unused/complete with tags/labels” → generate return instructions + address + label details → log reason code → send confirmation with deadline reminders.
- **Grounding:** 7-day policy; unused condition with tags/labels; returns must be posted within 7 days.

[Returns, Refund & Cancellation Policy](https://grest.in/pages/returns-refund-cancellation?srsltid=AfmBOopTktgZ4kgRat79dRwPk9wsnRGv17XDL6NSm_c9P74TuzctOoLx&utm_source=chatgpt.com)

1. **Order status & live tracking handoff**
    - **Trigger:** “Where is my order?”
    - **Flow:** Accept order ID/phone/email → fetch carrier/tracking → present ETA and last scan → offer SMS/email updates.
    - **Grounding:** Tracking ID promised post-dispatch; typical 24–48h processing + 4–6 business days delivery.
    
    [Shipping Policy](https://grest.in/pages/shipping)
    
2. **COD eligibility & EMI/BNPL pre-check**
    - **Trigger:** “Can I pay COD/EMI?”
    - **Flow:** Take pincode and cart → check COD eligibility; offer EMI options (incl. Bajaj) and BNPL → calculate installment schedule → add to checkout with chosen plan.
    - **Grounding:** COD available, check at checkout; EMI and BNPL supported.
    
    [Refurbished iPhone Q&A | Get Answers Here | GREST](https://grest.in/pages/faqs?srsltid=AfmBOorPmh7GZ-_gU7F7ePae4J8fOGyNi7VoFPQ9BSRgQNIzDd0x7oym)
    

### **7. Variant match & stock-aware product finder**

- **Trigger:** “I need an iPhone 13, 128GB, Good grade, under ₹27k.”
    - **Flow:** Query catalog by model/storage/grade/price → return live SKUs with “Buy” links → allow swap (e.g., Good→Fair) with savings delta → capture cart action.
    - **Grounding:** Live collections list models, storage, grades, pricing, and Buy CTAs.
        
        [Mobile Phones Collections- Grest.in](https://grest.in/collections/mobile-phones?srsltid=AfmBOop0ZxHpZehCWq_9Pk3SjjZJq3Uk92d9N5Ej4mU81TvJacJzOWzZ)
        
1. **Extended warranty upsell at purchase (or post-purchase add-on within window)**
    - **Trigger:** During checkout or post-purchase within X days.
    - **Flow:** Explain +6-month add-on at ₹1,499 → add to order or create separate add-on order → issue updated invoice.
    - **Grounding:** FAQ states extended 6-month warranty for ₹1,499. [Grest](https://grest.in/pages/faqs?srsltid=AfmBOorPmh7GZ-_gU7F7ePae4J8fOGyNi7VoFPQ9BSRgQNIzDd0x7oym)
2. **First-run device setup & health checklist (post-delivery concierge)**
    - **Trigger:** “How do I set it up?” or automated 24h after delivery.
    - **Flow:** Step-through SIM/iCloud/login, battery calibration tips, quick diagnostics (speaker/camera mic tests), and warranty-covered issue paths.
    - **Grounding:** FAQ lists troubleshooting coverage and what’s covered by warranty. [Grest](https://grest.in/pages/faqs?srsltid=AfmBOorPmh7GZ-_gU7F7ePae4J8fOGyNi7VoFPQ9BSRgQNIzDd0x7oym)
3. **Issue triage → auto-route to warranty vs non-warranty**
    - **Trigger:** “I dropped it; screen cracked.”
    - **Flow:** Classify as accidental/physical (not covered) → offer paid repair options or accessories; if functional defect, route to warranty claim flow.
    - **Grounding:** Non-coverage for accidental/liquid/display damage; functional issues covered. [Grest](https://grest.in/pages/warranty)
4. **Fraud-protection concierge (brand-protection workflow)**
    - **Trigger:** “I saw grest-store.in—is that you?”
    - **Flow:** Verify official domain(s) → warn about imposters → provide safe-purchase checklist → capture report.
    - **Grounding:** Grest blog warns about imposter site grest-store.in. [Grest](https://grest.in/blogs/news/beware-of-imposters-protecting-our-customers?srsltid=AfmBOoq43J8Tt5VJlzivNi98C7CfdHZzHM7O1kdGnHhGm0vVHFDJyZbj&utm_source=chatgpt.com)
5. **Accessory/parts compatibility advisor (post-sale attach)**
    - **Trigger:** “Will this charger/cable work?”
    - **Flow:** Confirm model → recommend compatible charger/cable and add to cart → share safety note.
    - **Grounding:** FAQ mentions compatible charging accessories and safety testing. [Grest](https://grest.in/pages/faqs?srsltid=AfmBOorPmh7GZ-_gU7F7ePae4J8fOGyNi7VoFPQ9BSRgQNIzDd0x7oym)
6. **Delivery-exception helpdesk (wrong/damaged item window) - DOA**
    - **Trigger:** “Received wrong/ damaged item.”
    - **Flow:** Validate 12-hour reporting window → collect photos/video → create replacement/refund case → logistics pickup coordination.
    - **Grounding:** FAQ sets a 12-hour window to report damage/incorrect delivery. [Grest](https://grest.in/pages/faqs?srsltid=AfmBOorPmh7GZ-_gU7F7ePae4J8fOGyNi7VoFPQ9BSRgQNIzDd0x7oym)
7. **Pan-India service/warranty location reassurance + pickup scheduling**
    - **Trigger:** “I’m in a non-metro—can you service?”
    - **Flow:** Confirm all-India validity → arrange pickup/return logistics for service.
    - **Grounding:** Warranty valid across India; service team guides pickup/repair/replacement. [Grest](https://grest.in/pages/faqs?srsltid=AfmBOorPmh7GZ-_gU7F7ePae4J8fOGyNi7VoFPQ9BSRgQNIzDd0x7oym)
8. **Stock alert & waitlist for specific grade/color/storage**
    - **Trigger:** Out-of-stock variant on PDP.
    - **Flow:** Capture desired variant → send alert when in stock or propose nearest alternatives (e.g., different grade/price).
    - **Grounding:** Live variant grids with grades/colors; some show “Pickup not available,” indicating fluctuating stock. [Grest](https://grest.in/collections/mobile-phones?srsltid=AfmBOop0ZxHpZehCWq_9Pk3SjjZJq3Uk92d9N5Ej4mU81TvJacJzOWzZ)
9. **Pre-purchase advisory vs “Superb/Good/Fair” trade-offs**
    - **Trigger:** “Is Superb worth it over Good?”
    - **Flow:** Explain grade definitions and battery thresholds; show price/performance delta across current SKUs; convert to cart.
    - **Grounding:** Grade definitions and battery health thresholds provided. [Grest](https://grest.in/pages/faqs?srsltid=AfmBOorPmh7GZ-_gU7F7ePae4J8fOGyNi7VoFPQ9BSRgQNIzDd0x7oym)
10. **Order modification window (address/phone) before dispatch**
    - **Trigger:** “Change my address.”
    - **Flow:** Check if order processed/dispatched (24–48h window); if eligible, update shipping info; if not, advise carrier redirect options.
    - **Grounding:** Processing time and courier handoff windows. [Grest](https://grest.in/pages/shipping)
11. **Post-purchase upsell: cases/screen protectors/battery-care tips**
    - **Trigger:** After delivery.
    - **Flow:** Offer protective add-ons and battery-care guidance; keep within warranty boundaries.
    - **Grounding:** Warranty coverage vs accidental damage helps frame upsell without over-promising. [Grest](https://grest.in/pages/warranty)
12. **“Track your order” self-serve widget embed**
    - **Trigger:** Site or WhatsApp bot.
    - **Flow:** User enters order ID/phone/email → bot fetches and shows status; optional notifications.
    - **Grounding:** FAQ promises tracking ID and live updates; site footer has “Track Your Order.” [Grest+1](https://grest.in/pages/faqs?srsltid=AfmBOorPmh7GZ-_gU7F7ePae4J8fOGyNi7VoFPQ9BSRgQNIzDd0x7oym)
13. **Escalation to human support with context bundle**
    - **Trigger:** Any failure path.
    - **Flow:** Hand off to care@grest.in or phone with full context (user, order, issue summary, artifacts like videos/photos).
    - **Grounding:** Official care contact info. [Grest](https://grest.in/collections/mobile-phones?srsltid=AfmBOop0ZxHpZehCWq_9Pk3SjjZJq3Uk92d9N5Ej4mU81TvJacJzOWzZ)