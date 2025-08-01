#!/usr/bin/env python3
"""
MySQL Schema Fetcher - Main Application with Report Type Filtering
Extracts, processes, and maps database schema for AI/LLM consumption
"""

import sys
import argparse
from pathlib import Path
from typing import Optional

from src.database.schema_extractor import schema_extractor, REPORT_TYPE_TABLES
from src.processors.etl_processor import etl_processor
from src.processors.schema_mapper import schema_mapper
from src.utils.config import config
from src.utils.logger import schema_logger, console

def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(
        description="MySQL Schema Fetcher - Extract and process database schema for AI/LLM with report filtering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --extract --report-type incident_details    # Extract incident tables only
  %(prog)s --process --report-type incident_details    # Process incident schema
  %(prog)s --full --report-type incident_details       # Complete pipeline for incidents
  %(prog)s --full --report-type all                    # Process all tables
  %(prog)s --list-report-types                         # Show available report types
        """
    )
    
    # Operation modes
    parser.add_argument(
        '--extract', 
        action='store_true',
        help='Extract raw schema from database'
    )
    
    parser.add_argument(
        '--process', 
        action='store_true',
        help='Process raw schema for AI/LLM consumption'
    )
    
    parser.add_argument(
        '--map', 
        action='store_true',
        help='Create natural language mappings'
    )
    
    parser.add_argument(
        '--full', 
        action='store_true',
        help='Run complete pipeline (extract + process + map)'
    )
    
    parser.add_argument(
        '--report', 
        action='store_true',
        help='Generate schema analysis report'
    )
    
    parser.add_argument(
        '--list-report-types',
        action='store_true',
        help='List available report types and their tables'
    )
    
    # Report type selection
    parser.add_argument(
        '--report-type',
        choices=list(REPORT_TYPE_TABLES.keys()),
        default='incident_details',
        help='Process specific report type tables only (default: incident_details)'
    )
    
    # Input/Output options
    parser.add_argument(
        '--raw-schema', 
        type=Path,
        help='Path to raw schema JSON file (for process/map operations)'
    )
    
    parser.add_argument(
        '--processed-schema', 
        type=Path,
        help='Path to processed schema JSON file (for map operation)'
    )
    
    parser.add_argument(
        '--output-dir', 
        type=Path,
        help='Output directory for generated files'
    )
    
    # Configuration options
    parser.add_argument(
        '--no-sample-data', 
        action='store_true',
        help='Skip extracting sample data from tables'
    )
    
    parser.add_argument(
        '--sample-rows', 
        type=int, 
        default=5,
        help='Number of sample rows to extract (default: 5)'
    )
    
    parser.add_argument(
        '--no-foreign-keys', 
        action='store_true',
        help='Skip extracting foreign key information'
    )
    
    parser.add_argument(
        '--no-indexes', 
        action='store_true',
        help='Skip extracting index information'
    )
    
    parser.add_argument(
        '--verbose', '-v', 
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Handle list report types
    if args.list_report_types:
        list_report_types()
        return 0
    
    # Validate arguments
    if not any([args.extract, args.process, args.map, args.full, args.report]):
        parser.error("Must specify at least one operation: --extract, --process, --map, --full, or --report")
    
    # Update configuration based on arguments
    if args.output_dir:
        config.processing.output_dir = args.output_dir
        config.processing.output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.no_sample_data:
        config.processing.include_sample_data = False
    
    if args.sample_rows:
        config.processing.sample_rows_limit = args.sample_rows
    
    if args.no_foreign_keys:
        config.processing.include_foreign_keys = False
    
    if args.no_indexes:
        config.processing.include_indexes = False
    
    if args.verbose:
        config.processing.log_level = "DEBUG"
    
    try:
        # Welcome message
        console.print("\n[bold blue]🗄️  MySQL Schema Fetcher v1.0.0[/bold blue]")
        console.print(f"[dim]Output directory: {config.processing.output_dir}[/dim]")
        console.print(f"[dim]Report type: {args.report_type}[/dim]\n")
        
        # Show report type info
        if args.report_type != "all":
            show_report_type_info(args.report_type)
        
        # Execute requested operations
        if args.full:
            run_full_pipeline(args.report_type)
        else:
            if args.extract:
                run_extraction(args.report_type)
            
            if args.process:
                run_processing(args.raw_schema, report_type=args.report_type)
            
            if args.map:
                run_mapping(args.processed_schema, report_type=args.report_type)
            
            if args.report:
                generate_reports(args.report_type)
        
        # Final summary
        schema_logger.log_section("OPERATION COMPLETED SUCCESSFULLY")
        console.print("[bold green]✅ All operations completed successfully![/bold green]")
        console.print(f"[dim]Output files saved to: {config.processing.output_dir}[/dim]\n")
        
        return 0
        
    except KeyboardInterrupt:
        schema_logger.log_warning("Operation cancelled by user")
        return 1
    except Exception as e:
        schema_logger.log_error(f"Operation failed: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

def list_report_types():
    """List available report types and their table mappings"""
    console.print("\n[bold blue]📋 Available Report Types[/bold blue]\n")
    
    for report_type, tables in REPORT_TYPE_TABLES.items():
        if report_type == "all":
            console.print(f"[bold green]• {report_type}[/bold green]")
            console.print("  [dim]Description: Process all database tables[/dim]")
            console.print("  [dim]Tables: All available tables in database[/dim]\n")
        else:
            console.print(f"[bold green]• {report_type}[/bold green]")
            console.print(f"  [dim]Description: {report_type.replace('_', ' ').title()} reporting tables[/dim]")
            console.print(f"  [dim]Table count: {len(tables)}[/dim]")
            console.print(f"  [dim]Tables: {', '.join(tables[:5])}{'...' if len(tables) > 5 else ''}[/dim]\n")

def show_report_type_info(report_type: str):
    """Show information about the selected report type"""
    info = schema_extractor.get_report_type_info(report_type)
    
    if "error" in info:
        console.print(f"[red]❌ {info['error']}[/red]")
        return
    
    console.print(f"[bold cyan]🎯 Report Type: {report_type}[/bold cyan]")
    console.print(f"[dim]Tables to process: {info['table_count']}[/dim]")
    
    if isinstance(info['tables'], list):
        console.print(f"[dim]Target tables: {', '.join(info['tables'])[:100]}{'...' if len(', '.join(info['tables'])) > 100 else ''}[/dim]\n")
    else:
        console.print(f"[dim]Target tables: {info['tables']}[/dim]\n")

def run_full_pipeline(report_type: str = "incident_details"):
    """Run the complete schema extraction and processing pipeline"""
    schema_logger.log_section("RUNNING FULL PIPELINE")
    
    # Step 1: Extract raw schema
    console.print(f"[bold cyan]Step 1/3:[/bold cyan] Extracting database schema for '{report_type}'...")
    raw_schema = run_extraction(report_type)
    
    # Step 2: Process schema
    console.print(f"\n[bold cyan]Step 2/3:[/bold cyan] Processing schema for AI/LLM...")
    processed_schema = run_processing(schema_data=raw_schema, report_type=report_type)
    
    # Step 3: Create mappings
    console.print(f"\n[bold cyan]Step 3/3:[/bold cyan] Creating natural language mappings...")
    run_mapping(schema_data=processed_schema, report_type=report_type)
    
    # Generate reports
    console.print(f"\n[bold cyan]Bonus:[/bold cyan] Generating analysis reports...")
    generate_reports(report_type)
    
    return True

def run_extraction(report_type: str = "incident_details"):
    """Extract raw schema from database"""
    try:
        # Extract schema with report type filter
        raw_schema = schema_extractor.extract_complete_schema(report_type=report_type)
        
        # Save raw schema
        raw_path = schema_extractor.save_raw_schema()
        
        # Generate and save extraction report
        report = schema_extractor.generate_schema_report()
        
        # Include report type in filename
        if report_type != "all":
            report_filename = f"extraction_report_{report_type}.txt"
        else:
            report_filename = "extraction_report.txt"
        
        report_path = config.get_output_path(report_filename)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        schema_logger.log_success(f"Extraction report saved to: {report_path}")
        
        return raw_schema
        
    except Exception as e:
        schema_logger.log_error(f"Schema extraction failed: {str(e)}")
        raise

def run_processing(schema_path: Optional[Path] = None, schema_data: Optional[dict] = None, report_type: str = "incident_details"):
    """Process raw schema for AI/LLM consumption"""
    try:
        # Load or use provided schema
        if schema_data:
            processed_schema = etl_processor.process_schema(schema_data)
        elif schema_path:
            etl_processor.load_raw_schema(schema_path)
            processed_schema = etl_processor.process_schema()
        else:
            # Try to load from default location with report type
            if report_type != "all":
                default_path = config.get_output_path(f"raw_schema_{report_type}.json")
            else:
                default_path = config.get_output_path("raw_schema.json")
            
            if default_path.exists():
                etl_processor.load_raw_schema(default_path)
                processed_schema = etl_processor.process_schema()
            else:
                raise FileNotFoundError(f"Raw schema file not found: {default_path}")
        
        # Save processed schema with report type in filename
        if report_type != "all":
            processed_filename = f"processed_schema_{report_type}.json"
        else:
            processed_filename = "processed_schema.json"
        
        processed_path = config.get_output_path(processed_filename)
        etl_processor.save_processed_schema(processed_path)
        
        return processed_schema
        
    except Exception as e:
        schema_logger.log_error(f"Schema processing failed: {str(e)}")
        raise

def run_mapping(schema_path: Optional[Path] = None, schema_data: Optional[dict] = None, report_type: str = "incident_details"):
    """Create natural language mappings"""
    try:
        # Load or use provided schema
        if schema_data:
            mappings = schema_mapper.create_mappings(schema_data)
        elif schema_path:
            schema_mapper.load_processed_schema(schema_path)
            mappings = schema_mapper.create_mappings()
        else:
            # Try to load from default location with report type
            if report_type != "all":
                default_path = config.get_output_path(f"processed_schema_{report_type}.json")
            else:
                default_path = config.get_output_path("processed_schema.json")
            
            if default_path.exists():
                schema_mapper.load_processed_schema(default_path)
                mappings = schema_mapper.create_mappings()
            else:
                raise FileNotFoundError(f"Processed schema file not found: {default_path}")
        
        # Save mappings with report type in filename
        if report_type != "all":
            mappings_filename = f"schema_mappings_{report_type}.json"
        else:
            mappings_filename = "schema_mappings.json"
        
        mappings_path = config.get_output_path(mappings_filename)
        schema_mapper.save_mappings(mappings_path)
        
        # Generate and save mapping summary
        summary = schema_mapper.generate_mapping_summary()
        
        if report_type != "all":
            summary_filename = f"mapping_summary_{report_type}.txt"
        else:
            summary_filename = "mapping_summary.txt"
        
        summary_path = config.get_output_path(summary_filename)
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary)
        
        schema_logger.log_success(f"Mapping summary saved to: {summary_path}")
        
        return mappings
        
    except Exception as e:
        schema_logger.log_error(f"Schema mapping failed: {str(e)}")
        raise

def generate_reports(report_type: str = "incident_details"):
    """Generate comprehensive analysis reports"""
    try:
        schema_logger.log_info("Generating comprehensive analysis reports...")
        
        # Determine filenames based on report type
        if report_type != "all":
            raw_file = f"raw_schema_{report_type}.json"
            processed_file = f"processed_schema_{report_type}.json"
            mappings_file = f"schema_mappings_{report_type}.json"
            extraction_report_file = f"extraction_report_{report_type}.txt"
            mapping_summary_file = f"mapping_summary_{report_type}.txt"
            combined_report_file = f"complete_analysis_{report_type}.txt"
            inventory_file = f"file_inventory_{report_type}.txt"
        else:
            raw_file = "raw_schema.json"
            processed_file = "processed_schema.json"
            mappings_file = "schema_mappings.json"
            extraction_report_file = "extraction_report.txt"
            mapping_summary_file = "mapping_summary.txt"
            combined_report_file = "complete_analysis.txt"
            inventory_file = "file_inventory.txt"
        
        # Check which files exist
        files_exist = {
            'raw': config.get_output_path(raw_file).exists(),
            'processed': config.get_output_path(processed_file).exists(),
            'mappings': config.get_output_path(mappings_file).exists()
        }
        
        reports_generated = []
        
        # Generate extraction report if raw schema exists
        if files_exist['raw']:
            try:
                if not schema_extractor.raw_schema:
                    import json
                    with open(config.get_output_path(raw_file), 'r') as f:
                        schema_extractor.raw_schema = json.load(f)
                
                report = schema_extractor.generate_schema_report()
                report_path = config.get_output_path(extraction_report_file)
                
                with open(report_path, 'w', encoding='utf-8') as f:
                    f.write(report)
                
                reports_generated.append(str(report_path))
            except Exception as e:
                schema_logger.log_warning(f"Could not generate extraction report: {str(e)}")
        
        # Generate mapping summary if mappings exist
        if files_exist['mappings']:
            try:
                if not schema_mapper.mappings:
                    import json
                    with open(config.get_output_path(mappings_file), 'r') as f:
                        schema_mapper.mappings = json.load(f)
                
                summary = schema_mapper.generate_mapping_summary()
                summary_path = config.get_output_path(mapping_summary_file)
                
                with open(summary_path, 'w', encoding='utf-8') as f:
                    f.write(summary)
                
                reports_generated.append(str(summary_path))
            except Exception as e:
                schema_logger.log_warning(f"Could not generate mapping summary: {str(e)}")
        
        # Generate combined analysis report
        combined_report = generate_combined_report(files_exist, report_type)
        combined_path = config.get_output_path(combined_report_file)
        
        with open(combined_path, 'w', encoding='utf-8') as f:
            f.write(combined_report)
        
        reports_generated.append(str(combined_path))
        
        # Generate file inventory
        inventory = generate_file_inventory(report_type)
        inventory_path = config.get_output_path(inventory_file)
        
        with open(inventory_path, 'w', encoding='utf-8') as f:
            f.write(inventory)
        
        reports_generated.append(str(inventory_path))
        
        schema_logger.log_success(f"Generated {len(reports_generated)} analysis reports")
        for report_path in reports_generated:
            schema_logger.log_info(f"  📄 {report_path}")
        
    except Exception as e:
        schema_logger.log_error(f"Report generation failed: {str(e)}")
        raise

def generate_combined_report(files_exist: dict, report_type: str) -> str:
    """Generate a combined analysis report"""
    from datetime import datetime
    
    report = []
    report.append("="*100)
    report.append(f"MYSQL SCHEMA FETCHER - COMPLETE ANALYSIS REPORT ({report_type.upper()})")
    report.append("="*100)
    report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Database: {config.database.database}")
    report.append(f"Host: {config.database.host}:{config.database.port}")
    report.append(f"Report Type: {report_type}")
    report.append("")
    
    # Configuration summary
    report.append("CONFIGURATION SUMMARY:")
    report.append("-" * 50)
    report.append(f"Output Directory: {config.processing.output_dir}")
    report.append(f"Report Type Filter: {report_type}")
    report.append(f"Include Sample Data: {config.processing.include_sample_data}")
    report.append(f"Sample Rows Limit: {config.processing.sample_rows_limit}")
    report.append(f"Include Foreign Keys: {config.processing.include_foreign_keys}")
    report.append(f"Include Indexes: {config.processing.include_indexes}")
    report.append("")
    
    # Files generated
    report.append("FILES GENERATED:")
    report.append("-" * 50)
    output_dir = config.processing.output_dir
    
    for file_path in output_dir.glob("*"):
        if file_path.is_file():
            # Only show files related to this report type
            if report_type != "all" and report_type not in file_path.name and "complete_analysis" not in file_path.name and "file_inventory" not in file_path.name:
                continue
            file_size = file_path.stat().st_size
            report.append(f"  📄 {file_path.name} ({file_size:,} bytes)")
    
    report.append("")
    
    # Processing summary
    if files_exist['raw'] or files_exist['processed'] or files_exist['mappings']:
        report.append("PROCESSING SUMMARY:")
        report.append("-" * 50)
        
        if files_exist['raw']:
            report.append("✅ Raw schema extraction completed")
        else:
            report.append("❌ Raw schema extraction not performed")
        
        if files_exist['processed']:
            report.append("✅ Schema processing completed")
        else:
            report.append("❌ Schema processing not performed")
        
        if files_exist['mappings']:
            report.append("✅ Natural language mapping completed")
        else:
            report.append("❌ Natural language mapping not performed")
        
        report.append("")
    
    # Next steps recommendations
    report.append("NEXT STEPS / RECOMMENDATIONS:")
    report.append("-" * 50)
    
    if not files_exist['raw']:
        report.append(f"1. Run schema extraction: python main.py --extract --report-type {report_type}")
    elif not files_exist['processed']:
        report.append(f"1. Process extracted schema: python main.py --process --report-type {report_type}")
    elif not files_exist['mappings']:
        report.append(f"1. Create natural language mappings: python main.py --map --report-type {report_type}")
    else:
        report.append("1. ✅ All processing steps completed!")
        report.append(f"2. Use processed_schema_{report_type}.json for LLM/AI applications")
        report.append(f"3. Use schema_mappings_{report_type}.json for natural language query processing")
        report.append("4. Integrate with your chatbot/LLM pipeline")
    
    report.append("")
    report.append("For more information, see individual report files and documentation.")
    report.append("="*100)
    
    return "\n".join(report)

def generate_file_inventory(report_type: str) -> str:
    """Generate an inventory of all output files"""
    from datetime import datetime
    
    inventory = []
    inventory.append(f"FILE INVENTORY - MySQL Schema Fetcher Output ({report_type.upper()})")
    inventory.append("="*60)
    inventory.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    inventory.append(f"Directory: {config.processing.output_dir}")
    inventory.append(f"Report Type: {report_type}")
    inventory.append("")
    
    output_dir = config.processing.output_dir
    
    # Categorize files
    file_categories = {
        "Schema Data": [".json"],
        "Reports": [".txt", ".md"],
        "Logs": [".log"],
        "Other": []
    }
    
    files_by_category = {category: [] for category in file_categories}
    
    for file_path in sorted(output_dir.glob("*")):
        if file_path.is_file():
            # Only include files related to this report type
            if report_type != "all":
                if report_type not in file_path.name and not any(x in file_path.name for x in ["complete_analysis", "file_inventory"]):
                    continue
            
            file_info = {
                "name": file_path.name,
                "size": file_path.stat().st_size,
                "modified": datetime.fromtimestamp(file_path.stat().st_mtime),
                "path": str(file_path)
            }
            
            # Categorize file
            categorized = False
            for category, extensions in file_categories.items():
                if extensions and file_path.suffix.lower() in extensions:
                    files_by_category[category].append(file_info)
                    categorized = True
                    break
            
            if not categorized:
                files_by_category["Other"].append(file_info)
    
    # Generate inventory by category
    for category, files in files_by_category.items():
        if files:
            inventory.append(f"{category.upper()}:")
            inventory.append("-" * 30)
            
            for file_info in files:
                size_str = f"{file_info['size']:,} bytes"
                modified_str = file_info['modified'].strftime('%Y-%m-%d %H:%M')
                inventory.append(f"  📄 {file_info['name']}")
                inventory.append(f"     Size: {size_str}")
                inventory.append(f"     Modified: {modified_str}")
                inventory.append("")
    
    # Summary statistics
    total_files = sum(len(files) for files in files_by_category.values())
    total_size = sum(
        file_info['size'] 
        for files in files_by_category.values() 
        for file_info in files
    )
    
    inventory.append("SUMMARY:")
    inventory.append("-" * 30)
    inventory.append(f"Total Files: {total_files}")
    inventory.append(f"Total Size: {total_size:,} bytes ({total_size/1024/1024:.2f} MB)")
    inventory.append("")
    
    # File descriptions
    file_descriptions = {
        f"raw_schema_{report_type}.json": f"Complete database schema for {report_type} report",
        f"processed_schema_{report_type}.json": f"Schema processed and optimized for AI/LLM ({report_type})",
        f"schema_mappings_{report_type}.json": f"Natural language mappings for {report_type} queries",
        f"extraction_report_{report_type}.txt": f"Detailed report of the schema extraction process ({report_type})",
        f"mapping_summary_{report_type}.txt": f"Summary of created natural language mappings ({report_type})",
        f"complete_analysis_{report_type}.txt": f"Combined analysis report with recommendations ({report_type})",
        f"file_inventory_{report_type}.txt": f"This file - inventory of all generated files ({report_type})",
        "schema_extraction.log": "Detailed logs from the extraction process"
    }
    
    inventory.append("FILE DESCRIPTIONS:")
    inventory.append("-" * 30)
    
    for file_info in sum(files_by_category.values(), []):
        description = file_descriptions.get(file_info['name'], "Generated output file")
        inventory.append(f"{file_info['name']}: {description}")
    
    inventory.append("")
    inventory.append("="*60)
    
    return "\n".join(inventory)

if __name__ == "__main__":
    sys.exit(main())