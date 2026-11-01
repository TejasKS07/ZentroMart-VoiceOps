"""Conditional calculations from caller-supplied facts, never private order data."""
from datetime import date, datetime
import re

from ..config import REFERENCE_DATE
from ..redaction import normalize_digits


def dates_in_text(text: str) -> set[date]:
    """Recognize explicit ISO/English calendar dates without inventing a day."""
    text=normalize_digits(text)
    result=set()
    for value in re.findall(r'\b\d{4}-\d{2}-\d{2}\b',text):
        try:result.add(date.fromisoformat(value))
        except ValueError:pass
    for day,month,year in re.findall(r'\b(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]+)(?:\s+(20\d{2}))?\b',text,re.I):
        for pattern in ('%d %B %Y','%d %b %Y'):
            try:result.add(datetime.strptime(f'{day} {month} {year or REFERENCE_DATE.year}',pattern).date());break
            except ValueError:pass
    return result


def without_echoed_dates(text,caller_dates):
    """Exclude only recognized dates explicitly stated by the caller.

    Other quantities still need policy/calculation support: a supplied date never
    licenses a made-up fee, duration or amount that happens to share its digits.
    """
    pattern=r'\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]+(?:\s+20\d{2})?\b'
    def replace(match):
        dates=dates_in_text(match[0])
        return '[caller date]' if dates and dates<=caller_dates else match[0]
    return re.sub(pattern,replace,normalize_digits(text),flags=re.I)


def complete_catalog_warranty(reply,question,evidence,language):
    """Include the unambiguous catalog duration in a warranty answer.

    This is a structured public lookup, not a question/answer map. The raw model
    trace is retained separately; an incorrect explicit duration is rejected.
    """
    if not re.search(r'warranty|assured|वारंटी|वारण्टी|गारंटी',question,re.I):return reply
    products=[e for e in evidence if e.metadata.get('doc_type')=='catalog']
    if len(products)!=1:return reply
    row=dict(line.split(': ',1) for line in products[0].text.splitlines() if ': ' in line)
    value=row.get('warranty_months','')
    if not value.isdigit() or int(value)<=0:return reply
    months=int(value)
    durations=re.findall(r'\b(\d+)\s*[- ]?\s*(months?|years?|mahine|saal|महीने|माह|साल|वर्ष)',normalize_digits(reply),re.I)
    for count,unit in durations:
        duration=int(count)*(12 if unit.lower() in {'year','years','saal','साल','वर्ष'} else 1)
        if duration!=months:raise ValueError('Warranty duration contradicts catalog')
    if durations:return reply
    note=(f'कैटलॉग में {months} महीने की वारंटी दर्ज है।' if language=='hi' else
        f'Catalog ke mutabik warranty {months} mahine ki hai.' if language=='hi-en' else
        f'The catalog lists a {months}-month warranty.')
    return reply.rstrip()+' '+note


def refund_calculations(text, calculator):
    """Only a single, explicit refund initiation date and payment method qualify.

    These are hypothetical caller facts; they never verify an order or prove that
    a refund actually happened. Ambiguous dates/methods are left for clarification.
    """
    if calculator is None or not re.search(r'refund|रिफंड|रिफ़ंड',text,re.I):return ()
    methods={name for name,pattern in (
        ('upi',r'\bupi\b|यूपीआई'),('card',r'\b(?:credit|debit)?\s*card\b|कार्ड'),
        ('netbanking',r'net\s*bank(?:ing)?|netbanking|नेट\s*बैंकिंग'),
        ('wallet',r'\bwallet\b|वॉलेट')) if re.search(pattern,text,re.I)}
    if len(methods)!=1:return ()
    initiation=re.search(r'(?:refund[^.!?\n]{0,60}(?:initiat\w*|started|शुरू)|(?:initiat\w*|started|शुरू)[^.!?\n]{0,40}refund)',text,re.I)
    if not initiation:return ()
    # Multiple dates could describe delivery, initiation or an earlier contact.
    dates=dates_in_text(text)
    if len(dates)!=1:return ()
    initiated=next(iter(dates))
    if initiated>calculator.today:return ()
    return (calculator.refund_timeline(next(iter(methods)),initiated),)
